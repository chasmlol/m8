//! Text-to-speech, from a server with Breeze TTS 2's API (start-breeze.ps1): it clones the voice
//! in a WAV, takes how to say each line ("shouting over gunfire") and sounds in it ("(laugh)"),
//! and streams the audio as it's made, which goes on to the game. Every speaker brings the clip
//! of their own voice. For another kind of TTS, `Voice::stream` is the one place to change.

use crate::game::Game;
use crate::site::Trace;
use crate::Speaking;
use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Work for the TTS thread, tagged with the `Speaking` generation it was produced for, so
/// lines queued before the player talked over the speaker get dropped.
pub enum SpeechJob {
    /// A sentence of a speaker's line, how to say it (may be empty), and the clip of their voice.
    Sentence { speaker: String, text: String, how: String, voice: PathBuf, generation: u64 },
    /// The speaker's line is finished.
    End { speaker: String, generation: u64 },
}

/// A voice clip and what's said in it.
struct Reference {
    wav: Vec<u8>,
    text: String,
}

pub struct Voice {
    url: String,
    agent: ureq::Agent,
    /// The clips it clones, by their files, read once.
    references: HashMap<PathBuf, Reference>,
    seed: u64,
    /// Say each line as its "[...]" note says (Breeze's guidance: a pass with the instruction and
    /// one without, blended, so about twice the work), or plainly.
    instruct: bool,
    pub sample_rate: u32,
}

/// How strongly Breeze follows an instruction (what its docs advise for directing a cloned
/// voice; without one it takes only 1). breeze_profile.json warms the server up for both.
const BREEZE_CFG: f32 = 4.0;

impl Voice {
    /// Connects to the voice server (its sample rate is every voice's).
    pub fn connect(url: &str, seed: u64) -> Result<Self> {
        let agent = ureq::AgentBuilder::new().build();
        let info: serde_json::Value = agent
            .get(&format!("{url}/health"))
            .call()
            .map_err(|e| anyhow!("voice server at {url} isn't up (start-breeze.ps1): {e}"))?
            .into_json()?;
        let sample_rate = info["sample_rate"].as_u64().ok_or_else(|| anyhow!("voice server sent no sample rate"))? as u32;
        Ok(Voice { url: url.to_string(), agent, references: HashMap::new(), seed, instruct: true, sample_rate })
    }

    /// Whether lines are said as their notes say (on unless turned off): off, Breeze runs twice as
    /// fast, which matters where the GPU is shared with a game's rendering.
    pub fn instructed(mut self, on: bool) -> Self {
        self.instruct = on;
        self
    }

    /// The clip at `path` and its transcript (the .txt beside it), read the first time it's used.
    fn reference(&mut self, path: &Path) -> Result<&Reference> {
        if !self.references.contains_key(path) {
            let read = |path: &Path| std::fs::read(path).map_err(|e| anyhow!("reading {}: {e}", path.display()));
            let wav = read(path)?;
            let text = String::from_utf8(read(&path.with_extension("txt"))?)?.trim().to_string();
            self.references.insert(path.to_path_buf(), Reference { wav, text });
        }
        Ok(&self.references[path])
    }

    /// Synthesizes `text` in the voice of the clip at `voice`, said as `how`, handing over audio
    /// as soon as each piece arrives. Stops early when `on_chunk` returns false (dropping the
    /// connection stops the server's generation).
    pub fn stream(&mut self, text: &str, how: &str, voice: &Path, mut on_chunk: impl FnMut(&[f32]) -> bool) -> Result<()> {
        const BOUNDARY: &str = "mate-speech-boundary";
        let (seed, instruct) = (self.seed, self.instruct);
        let reference = self.reference(voice)?;
        let mut body = Vec::new();
        let mut field = |name: &str, value: &str| {
            body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").bytes());
        };
        let how = if instruct { how } else { "" };
        field("text", text);
        if !how.is_empty() {
            field("instruction", how);
        }
        field("cfg_scale", &if how.is_empty() { 1.0 } else { BREEZE_CFG }.to_string());
        field("seed", &seed.to_string());
        field("ref_text", &reference.text);
        body.extend(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"ref_audio\"; filename=\"reference.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                .bytes(),
        );
        body.extend(&reference.wav);
        body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
        let request = || {
            self.agent
                .post(&format!("{}/v1/audio/speech", self.url))
                .set("Content-Type", &format!("multipart/form-data; boundary={BOUNDARY}"))
                .send_bytes(&body)
        };
        // (409: still finishing a line it was cut off in)
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

pub fn run(mut voice: Voice, game: Arc<dyn Game>, speaking: Speaking, trace: Trace, jobs: Receiver<SpeechJob>) -> Result<()> {
    let sample_rate = voice.sample_rate;
    // (the line whose audio went to the game last, and when that audio runs out: a new line
    // takes over from an older one still playing as its first audio arrives, not before)
    let mut audio_generation = None;
    let mut audio_until = Instant::now();
    // (a job taken while gathering a line's sentences, for next)
    let mut held = None;
    loop {
        let job = match held.take() {
            Some(job) => job,
            None => match jobs.recv() {
                Ok(job) => job,
                Err(_) => break,
            },
        };
        match job {
            SpeechJob::Sentence { generation, .. } if generation != speaking.generation() => {
                // Queued before the player talked over the speaker.
            }
            SpeechJob::End { speaker, generation } => {
                if generation == speaking.generation() {
                    game.line_end(&speaker);
                }
                speaking.end_line();
            }
            SpeechJob::Sentence { speaker, mut text, how, voice: clip, generation } => {
                // The line's next sentences already written go with it, as one request: each
                // request has a wait of its own before its first audio (a line's first sentence
                // goes alone, before the rest is written; the rest follow it as one).
                while let Ok(next) = jobs.try_recv() {
                    match next {
                        SpeechJob::Sentence { speaker: s, text: t, how: h, voice: v, generation: g }
                            if s == speaker && g == generation && v == clip && (h == how || !voice.instruct) =>
                        {
                            text.push(' ');
                            text.push_str(&t);
                        }
                        other => {
                            held = Some(other);
                            break;
                        }
                    }
                }
                game.sentence(&speaker, &text, &how);
                let start = Instant::now();
                let mut first_chunk_ms = None;
                let result = voice.stream(&text, &how, &clip, |samples| {
                    if speaking.generation() != generation {
                        return false;
                    }
                    first_chunk_ms.get_or_insert(start.elapsed().as_millis());
                    if audio_generation != Some(generation) {
                        if Instant::now() < audio_until {
                            game.stop_audio();
                        }
                        audio_generation = Some(generation);
                    }
                    let length = Duration::from_secs_f32(samples.len() as f32 / sample_rate as f32);
                    audio_until = audio_until.max(Instant::now()) + length;
                    speaking.extend(length);
                    game.audio(&speaker, samples, sample_rate);
                    true
                });
                if speaking.generation() == generation {
                    game.sentence_end(&speaker);
                }
                if let Err(e) = &result {
                    eprintln!("[tts] failed on \"{text}\": {e:#}");
                }
                trace.add(
                    "voice",
                    json!({
                        "speaker": speaker,
                        "text": text,
                        "how": how,
                        "voice": clip,
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
