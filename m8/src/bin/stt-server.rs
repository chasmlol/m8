//! `stt-server`: Parakeet TDT (sherpa-onnx's int8 export) behind an OpenAI-style speech-to-text
//! endpoint, for m8 (or anything) to call: POST /v1/audio/transcriptions with a WAV (a multipart
//! `file`, or the body itself) gives `{"text": ...}`; GET /health says it's up. With the form's
//! `response_format` set to `verbose_json` it also gives `words`, each with its `start` and `end`
//! in seconds (OpenAI's `timestamp_granularities[]=word`): when each word is said, which is when a
//! character's mouth makes it.
//!
//! Usage: `stt-server <the model's folder> [port]` (port 8083 if none).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::json;
use sherpa_onnx::{LinearResampler, OfflineRecognizer, OfflineRecognizerConfig};
use std::path::Path;
use std::time::Instant;

const SAMPLE_RATE: i32 = 16000;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(args.first().context("usage: stt-server <model folder> [port]")?);
    let port: u16 = args.get(1).map(|p| p.parse()).transpose()?.unwrap_or(8083);

    let file = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer.encoder = file("encoder.int8.onnx");
    config.model_config.transducer.decoder = file("decoder.int8.onnx");
    config.model_config.transducer.joiner = file("joiner.int8.onnx");
    config.model_config.tokens = file("tokens.txt");
    config.model_config.model_type = Some("nemo_transducer".to_string());
    config.model_config.num_threads = 4;
    let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| anyhow!("couldn't load the model in {}", dir.display()))?;

    let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow!("{e}"))?;
    println!("Speech to text on http://127.0.0.1:{port}/v1/audio/transcriptions");
    for mut request in server.incoming_requests() {
        let reply = match (request.method().as_str(), request.url().split('?').next().unwrap_or("")) {
            ("GET", "/health") => tiny_http::Response::from_string(json!({"status": "ok"}).to_string()),
            ("POST", "/v1/audio/transcriptions") => {
                let content_type = request
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("Content-Type"))
                    .map(|h| h.value.as_str().to_string())
                    .unwrap_or_default();
                let mut body = Vec::new();
                let _ = request.as_reader().read_to_end(&mut body);
                let start = Instant::now();
                let verbose = field(&content_type, &body, "response_format").is_some_and(|f| f.trim_ascii() == b"verbose_json");
                match wav_of(&content_type, &body).and_then(|wav| samples(&wav)) {
                    Ok(audio) => {
                        let stream = recognizer.create_stream();
                        stream.accept_waveform(SAMPLE_RATE, &audio);
                        recognizer.decode(&stream);
                        let result = stream.get_result();
                        let text = result.as_ref().map(|r| r.text.trim().to_string()).unwrap_or_default();
                        let seconds = audio.len() as f32 / SAMPLE_RATE as f32;
                        println!("\"{text}\" ({seconds:.1}s audio, {} ms)", start.elapsed().as_millis());
                        let reply = match (verbose, &result) {
                            (true, Some(result)) => json!({"text": text, "duration": seconds, "words": words(result, seconds)}),
                            _ => json!({"text": text}),
                        };
                        tiny_http::Response::from_string(reply.to_string())
                    }
                    Err(e) => tiny_http::Response::from_string(json!({"error": format!("{e:#}")}).to_string()).with_status_code(400),
                }
            }
            _ => tiny_http::Response::from_string("").with_status_code(404),
        };
        let _ = request.respond(reply.with_header("Content-Type: application/json".parse::<tiny_http::Header>().unwrap()));
    }
    Ok(())
}

/// The words heard and when each is said. Parakeet's tokens are pieces of words, each one that
/// starts a word beginning with a space, at the time it was heard; a word ends with its last
/// piece's duration, or where the next word begins.
fn words(result: &sherpa_onnx::OfflineRecognizerResult, seconds: f32) -> Vec<serde_json::Value> {
    let Some(times) = &result.timestamps else { return Vec::new() };
    let mut words: Vec<(String, f32, f32)> = Vec::new();
    for (i, (token, &at)) in result.tokens.iter().zip(times).enumerate() {
        let starts = token.starts_with([' ', '\u{2581}']) || words.is_empty();
        let piece = token.trim_start_matches([' ', '\u{2581}']);
        let end = result.durations.as_ref().and_then(|d| d.get(i)).map_or(seconds, |d| at + d.max(0.04));
        match words.last_mut() {
            Some(word) if !starts => {
                word.0.push_str(piece);
                word.2 = end;
            }
            _ => words.push((piece.to_string(), at, end)),
        }
    }
    for i in 0..words.len().saturating_sub(1) {
        words[i].2 = words[i].2.max(words[i].1).min(words[i + 1].1);
    }
    words
        .into_iter()
        .filter(|(word, ..)| word.chars().any(char::is_alphanumeric))
        .map(|(word, start, end)| json!({"word": word, "start": start, "end": end.min(seconds)}))
        .collect()
}

/// A part of a multipart form, by its name (none for a body that isn't a form).
fn field<'a>(content_type: &str, body: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let boundary = content_type.split("boundary=").nth(1)?;
    let marker = format!("--{}", boundary.trim_matches('"'));
    let find = |haystack: &[u8], needle: &[u8], from: usize| {
        haystack[from..].windows(needle.len()).position(|w| w == needle).map(|at| at + from)
    };
    let mut at = 0;
    while let Some(part) = find(body, marker.as_bytes(), at) {
        let headers_start = part + marker.len();
        let headers_end = find(body, b"\r\n\r\n", headers_start)?;
        let headers = String::from_utf8_lossy(&body[headers_start..headers_end]);
        let data_start = headers_end + 4;
        let data_end = find(body, format!("\r\n{marker}").as_bytes(), data_start).unwrap_or(body.len());
        if headers.contains(&format!("name=\"{name}\"")) {
            return Some(&body[data_start..data_end]);
        }
        at = data_end;
    }
    None
}

/// The WAV in a request: the multipart form's `file`, else the body itself.
fn wav_of<'a>(content_type: &str, body: &'a [u8]) -> Result<&'a [u8]> {
    if !content_type.contains("boundary=") {
        return Ok(body);
    }
    field(content_type, body, "file").ok_or_else(|| anyhow!("no `file` in the form"))
}

/// A WAV's audio (16-bit or float, any channels and rate) as 16 kHz mono.
fn samples(wav: &[u8]) -> Result<Vec<f32>> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        bail!("not a WAV");
    }
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 1u16, 0u32, 16u16);
    let mut at = 12;
    while at + 8 <= wav.len() {
        let id = &wav[at..at + 4];
        let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into()?) as usize;
        let data = &wav[at + 8..(at + 8 + size).min(wav.len())];
        if id == b"fmt " && data.len() >= 16 {
            format = u16::from_le_bytes([data[0], data[1]]);
            channels = u16::from_le_bytes([data[2], data[3]]).max(1);
            rate = u32::from_le_bytes(data[4..8].try_into()?);
            bits = u16::from_le_bytes([data[14], data[15]]);
        } else if id == b"data" {
            let frames: Vec<f32> = match (format, bits) {
                (1, 16) => data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect(),
                (3, 32) => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                _ => bail!("a WAV of 16-bit or float samples, please"),
            };
            let mono: Vec<f32> = frames
                .chunks(channels as usize)
                .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                .collect();
            return Ok(match LinearResampler::create(rate as i32, SAMPLE_RATE) {
                Some(resampler) if rate as i32 != SAMPLE_RATE => resampler.resample(&mono, true),
                _ => mono,
            });
        }
        at += 8 + size + size % 2;
    }
    bail!("no audio in the WAV")
}
