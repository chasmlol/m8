//! Text-to-speech, from a server with Breeze TTS 2's API (start-breeze.ps1): it clones the voice
//! in a WAV, takes how to say each line ("shouting over gunfire") and sounds in it ("(laugh)"),
//! and streams the audio as it's made, which goes on to the game. For another kind of TTS,
//! `Voice::stream` is the one place to change.

use crate::config::Config;
use crate::link::GameLink;
use crate::site::Trace;
use crate::Speaking;
use anyhow::{anyhow, Result};
use serde_json::json;
use std::io::Read;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

/// Work for the TTS thread, tagged with the `Speaking` generation it was produced for, so
/// lines queued before the player talked over the mate get dropped.
pub enum SpeechJob {
    /// A sentence, and how to say it (may be empty).
    Sentence(String, String, u64),
    /// The current reply is finished.
    End(u64),
}

pub struct Voice {
    url: String,
    agent: ureq::Agent,
    /// The WAV it clones, and what's said in it.
    reference: Vec<u8>,
    reference_text: String,
    seed: u64,
    pub sample_rate: u32,
}

/// How strongly Breeze follows an instruction (what its docs advise for directing a cloned
/// voice; without one it takes only 1). breeze_profile.json warms the server up for both.
const BREEZE_CFG: f32 = 4.0;

impl Voice {
    /// Connects to the voice server, and reads the clip it clones.
    pub fn load(cfg: &Config) -> Result<Self> {
        let url = cfg.mate.voice_server.clone();
        let read = |path: &std::path::Path| std::fs::read(path).map_err(|e| anyhow!("reading {}: {e}", path.display()));
        let reference = read(&cfg.mate.voice_reference)?;
        let reference_text = String::from_utf8(read(&cfg.mate.voice_reference.with_extension("txt"))?)?.trim().to_string();
        let agent = ureq::AgentBuilder::new().build();
        let info: serde_json::Value = agent
            .get(&format!("{url}/health"))
            .call()
            .map_err(|e| anyhow!("voice server at {url} isn't up (start-breeze.ps1): {e}"))?
            .into_json()?;
        let sample_rate = info["sample_rate"].as_u64().ok_or_else(|| anyhow!("voice server sent no sample rate"))? as u32;
        Ok(Voice { url, agent, reference, reference_text, seed: cfg.mate.voice_seed, sample_rate })
    }

    /// Synthesizes `text`, said as `how`, handing over audio as soon as each piece arrives.
    /// Stops early when `on_chunk` returns false (dropping the connection stops the server's
    /// generation).
    pub fn stream(&self, text: &str, how: &str, mut on_chunk: impl FnMut(&[f32]) -> bool) -> Result<()> {
        const BOUNDARY: &str = "mate-speech-boundary";
        let mut body = Vec::new();
        let mut field = |name: &str, value: &str| {
            body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").bytes());
        };
        field("text", text);
        if !how.is_empty() {
            field("instruction", how);
        }
        field("cfg_scale", &if how.is_empty() { 1.0 } else { BREEZE_CFG }.to_string());
        field("seed", &self.seed.to_string());
        field("ref_text", &self.reference_text);
        body.extend(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"ref_audio\"; filename=\"reference.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                .bytes(),
        );
        body.extend(&self.reference);
        body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
        let request = || {
            self.agent
                .post(&format!("{}/v1/audio/speech", self.url))
                .set("Content-Type", &format!("multipart/form-data; boundary={BOUNDARY}"))
                .send_bytes(&body)
        };
        // (409: still finishing a line he was cut off in)
        let mut response = request();
        for _ in 0..20 {
            if !matches!(response, Err(ureq::Error::Status(409, _))) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
            response = request();
        }
        let mut reader = response?.into_reader();
        // 16-bit samples; a read can end halfway through one.
        let (mut buffer, mut held) = (vec![0u8; 4800], 0);
        loop {
            let read = reader.read(&mut buffer[held..])?;
            if read == 0 {
                break;
            }
            let whole = (held + read) / 2 * 2;
            let samples: Vec<f32> = buffer[..whole]
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            buffer.copy_within(whole..held + read, 0);
            held = held + read - whole;
            if !on_chunk(&samples) {
                break;
            }
        }
        Ok(())
    }
}

pub fn run(voice: Voice, link: GameLink, speaking: Speaking, trace: Trace, jobs: Receiver<SpeechJob>) -> Result<()> {
    let sample_rate = voice.sample_rate as f32;
    // (the line whose audio went to the game last, and when that audio runs out: a new line
    // takes over from an older one still playing as its first audio arrives, not before)
    let mut audio_generation = None;
    let mut audio_until = Instant::now();
    for job in jobs {
        match job {
            SpeechJob::Sentence(_, _, generation) if generation != speaking.generation() => {
                // Queued before the player talked over the mate.
            }
            SpeechJob::End(generation) => {
                if generation == speaking.generation() {
                    link.send_json(&json!({"type": "audio_end"}));
                }
                speaking.end_line();
            }
            SpeechJob::Sentence(text, how, generation) => {
                link.say(&text);
                let start = Instant::now();
                let mut first_chunk_ms = None;
                let result = voice.stream(&text, &how, |samples| {
                    if speaking.generation() != generation {
                        return false;
                    }
                    first_chunk_ms.get_or_insert(start.elapsed().as_millis());
                    if audio_generation != Some(generation) {
                        if Instant::now() < audio_until {
                            link.send_json(&json!({"type": "stop_audio"}));
                        }
                        audio_generation = Some(generation);
                    }
                    audio_until = audio_until.max(Instant::now()) + Duration::from_secs_f32(samples.len() as f32 / sample_rate);
                    let pcm: Vec<u8> = samples
                        .iter()
                        .flat_map(|&s| ((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())
                        .collect();
                    speaking.extend(Duration::from_secs_f32(samples.len() as f32 / sample_rate));
                    link.send_audio(&pcm);
                    true
                });
                if let Err(e) = &result {
                    eprintln!("[tts] failed on \"{text}\": {e:#}");
                }
                trace.add(
                    "voice",
                    json!({
                        "text": text,
                        "how": how,
                        "first_audio_ms": first_chunk_ms.map(|ms| ms as u64),
                        "ms": start.elapsed().as_millis() as u64,
                        "cut_off": speaking.generation() != generation,
                        "error": result.as_ref().err().map(|e| format!("{e:#}")),
                    }),
                );
                println!(
                    "[tts] [{how}] \"{}\" (first audio {} ms, total {} ms)",
                    text,
                    first_chunk_ms.unwrap_or(0),
                    start.elapsed().as_millis()
                );
            }
        }
    }
    Ok(())
}
