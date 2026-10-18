//! `mate`: the voice brain for the Agentic NPC games (Minecraft, Halo).
//!
//! mic -> Silero VAD -> Parakeet (STT) -> LLM (words) -> Pocket TTS -> the game, then Kev (decide)
//! Watch and poke it at http://127.0.0.1:8090 (site.rs).
//!
//! Usage: `mate [config.toml]`, `mate [config.toml] --self-test`
//! or `mate [config.toml] --transcribe a.wav b.wav`.
//! While running you can also type a line into the console to "say" it without a mic.

mod brain;
mod config;
mod link;
mod listen;
mod picture;
mod site;
mod speak;

use anyhow::Result;
use config::Config;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Shared state about the mate's voice: when its queued speech will finish playing, and a
/// generation counter that the player bumps by talking over it. Everything produced for an
/// older generation (LLM tokens, TTS audio) is dropped.
#[derive(Clone)]
pub struct Speaking {
    until: Arc<Mutex<Instant>>,
    generation: Arc<AtomicU64>,
    /// Lines begun (the LLM writing them) whose voicing hasn't finished: speak.rs counts each
    /// off at its end.
    lines: Arc<AtomicUsize>,
}

impl Speaking {
    /// Extra time counted as "still speaking" after the audio, covering playback latency and room echo.
    const TAIL: Duration = Duration::from_millis(500);

    fn new() -> Self {
        Speaking {
            until: Arc::new(Mutex::new(Instant::now())),
            generation: Arc::new(AtomicU64::new(0)),
            lines: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// A line begun: until its end, he's talking.
    pub fn begin_line(&self) {
        self.lines.fetch_add(1, Ordering::SeqCst);
    }

    /// A line's voicing finished (or was dropped).
    pub fn end_line(&self) {
        let _ = self.lines.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
    }

    /// When his voice stops (or stopped): none while a line is still being written or voiced.
    pub fn quiet_from(&self) -> Option<Instant> {
        (self.lines.load(Ordering::SeqCst) == 0).then(|| *self.until.lock().unwrap())
    }

    pub fn extend(&self, audio: Duration) {
        let mut until = self.until.lock().unwrap();
        *until = (*until).max(Instant::now()) + audio;
    }

    pub fn is_speaking(&self) -> bool {
        Instant::now() < *self.until.lock().unwrap() + Self::TAIL
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// The player talked over the mate: abandon everything it was about to say.
    pub fn interrupt(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        *self.until.lock().unwrap() = Instant::now() - Self::TAIL;
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let config_path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("mate.toml"));
    let cfg = Config::load(&config_path)?;

    if args.iter().any(|a| a == "--self-test") {
        return self_test(&cfg);
    }
    if let Some(i) = args.iter().position(|a| a == "--simulate-mic") {
        // Plays WAVs through the live pipeline (48 kHz, 10 ms chunks -> resample -> VAD -> STT).
        let mut cfg = cfg.clone();
        cfg.mic.save_recordings = false;
        let mut listener = listen::Listener::new(cfg.clone(), listen::create_recognizer(&cfg)?, 48000)?;
        for path in &args[i + 1..] {
            let wave = sherpa_onnx::Wave::read(path).ok_or_else(|| anyhow::anyhow!("can't read {path}"))?;
            let mut audio = sherpa_onnx::LinearResampler::create(wave.sample_rate(), 48000)
                .map(|r| r.resample(wave.samples(), true))
                .unwrap_or_else(|| wave.samples().to_vec());
            audio.extend(std::iter::repeat_n(0.0, 48000 * 2));
            println!("== {path}");
            for chunk in audio.chunks(480) {
                for text in listener.feed(chunk.to_vec(), false) {
                    println!("   heard: \"{text}\"");
                }
            }
        }
        return Ok(());
    }
    if let Some(i) = args.iter().position(|a| a == "--transcribe") {
        let recognizer = listen::create_recognizer(&cfg)?;
        for path in &args[i + 1..] {
            let wave = sherpa_onnx::Wave::read(path).ok_or_else(|| anyhow::anyhow!("can't read {path}"))?;
            let samples = sherpa_onnx::LinearResampler::create(wave.sample_rate(), 16000)
                .map(|r| r.resample(wave.samples(), true))
                .unwrap_or_else(|| wave.samples().to_vec());
            println!("{path}: \"{}\"", listen::transcribe(&recognizer, &samples));
        }
        return Ok(());
    }

    println!("Loading speech models...");
    let recognizer = listen::create_recognizer(&cfg)?;
    let voice = speak::Voice::load(&cfg)?;

    let (inputs_tx, inputs_rx) = mpsc::channel::<brain::Input>();
    let (events_tx, events_rx) = mpsc::channel();
    let (hearing_tx, hearing_rx) = mpsc::channel::<link::Heard>();
    let hello = json!({"type": "hello", "name": cfg.mate.name, "sample_rate": voice.sample_rate});
    let link = link::GameLink::start(cfg.game.port, hello, events_tx, hearing_tx)?;
    println!("Waiting for a game on 127.0.0.1:{}", cfg.game.port);

    let forward = inputs_tx.clone();
    std::thread::spawn(move || {
        for msg in events_rx {
            let text = |key: &str| msg[key].as_str().unwrap_or_default().to_string();
            let input = match msg["type"].as_str() {
                Some("saved") => brain::Input::Saved(text("id")),
                Some("loaded") => brain::Input::Loaded(text("id")),
                // (typed in the game by the player, as a game master's line: what models are
                // trained to take as out of character, about the world)
                Some("direct") => brain::Input::Event { text: format!("Game master: {}", text("text")), react: true },
                _ => brain::Input::Event { text: text("text"), react: msg["react"].as_bool().unwrap_or(true) },
            };
            let _ = forward.send(input);
        }
    });

    // Dialogue the game plays (the radio, cutscenes) is transcribed into the feed, with a
    // recognizer of its own so that the mic's isn't held up.
    {
        let (cfg, inputs) = (cfg.clone(), inputs_tx.clone());
        std::thread::spawn(move || {
            let recognizer = match listen::create_recognizer(&cfg) {
                Ok(r) => r,
                Err(e) => return eprintln!("[hearing] no recognizer: {e:#}"),
            };
            for (about, samples) in hearing_rx {
                let started = Instant::now();
                let rate = about["sample_rate"].as_i64().unwrap_or(16000) as i32;
                let length = Duration::from_secs_f32(samples.len() as f32 / rate as f32);
                let samples = match sherpa_onnx::LinearResampler::create(rate, 16000) {
                    Some(r) if rate != 16000 => r.resample(&samples, true),
                    _ => samples,
                };
                let text = listen::transcribe(&recognizer, &samples);
                if text.is_empty() {
                    continue;
                }
                let speaker = about["speaker"].as_str().unwrap_or("Someone");
                let over_radio = if about["radio"].as_bool().unwrap_or(false) { " over the radio" } else { "" };
                println!("[hearing] {speaker}{over_radio}: \"{text}\"");
                // (the line is sent as it starts: he lets them finish before he answers)
                std::thread::sleep(length.saturating_sub(started.elapsed()));
                let _ = inputs.send(brain::Input::Event {
                    text: format!("{speaker}{over_radio}: \"{text}\""),
                    react: about["react"].as_bool().unwrap_or(true),
                });
            }
        });
    }

    let trace = site::Trace::new(&cfg.base_dir.join("logs"));
    let system = site::PromptFile::load(cfg.base_dir.join("system_prompt.txt"))?;
    let combat = site::PromptFile::load(cfg.base_dir.join("combat_prompt.txt"))?;
    let radio = site::PromptFile::load(cfg.base_dir.join("radio_prompt.txt"))?;
    {
        let (trace, system, combat, radio, link, inputs) =
            (trace.clone(), system.clone(), combat.clone(), radio.clone(), link.clone(), inputs_tx.clone());
        std::thread::spawn(move || {
            if let Err(e) = site::run(trace, system, combat, radio, link, inputs) {
                eprintln!("[site] stopped: {e:#}");
            }
        });
    }
    println!("Brain page: http://127.0.0.1:{}", site::PORT);

    let speaking = Speaking::new();
    let (speech_tx, speech_rx) = mpsc::channel();
    {
        let (link, speaking, trace) = (link.clone(), speaking.clone(), trace.clone());
        std::thread::spawn(move || {
            if let Err(e) = speak::run(voice, link, speaking, trace, speech_rx) {
                eprintln!("[tts] stopped: {e:#}");
            }
        });
    }

    {
        let (cfg, heard_tx, link, speaking) = (cfg.clone(), inputs_tx.clone(), link.clone(), speaking.clone());
        std::thread::spawn(move || {
            let (tx, rx) = mpsc::channel::<String>();
            std::thread::spawn(move || {
                for text in rx {
                    let _ = heard_tx.send(brain::Input::Heard(text));
                }
            });
            if let Err(e) = listen::run(cfg, recognizer, speaking, link, tx) {
                eprintln!("[mic] stopped: {e:#}");
            }
        });
    }

    // Typed lines count as speech, handy for testing without a mic.
    let brain_inputs = inputs_tx.clone();
    let typed = inputs_tx;
    std::thread::spawn(move || {
        for line in std::io::stdin().lines().map_while(Result::ok) {
            let line = line.trim().to_string();
            if !line.is_empty() {
                let _ = typed.send(brain::Input::Heard(line));
            }
        }
    });

    println!("Ready! Talk to {} (or type a line here).", cfg.mate.name);
    brain::Brain::new(cfg, link, speech_tx, speaking, trace, system, combat, radio, brain_inputs).run(inputs_rx);
    Ok(())
}

/// Loads every model and times STT and TTS without needing a mic or the game.
fn self_test(cfg: &Config) -> Result<()> {
    let t = Instant::now();
    let recognizer = listen::create_recognizer(cfg)?;
    println!("Parakeet loaded in {} ms", t.elapsed().as_millis());
    let wav_path = cfg.models.parakeet_dir.join("test_wavs").join("0.wav");
    let wave = sherpa_onnx::Wave::read(&config::path_str(&wav_path)).ok_or_else(|| anyhow::anyhow!("missing test wav"))?;
    let samples = if wave.sample_rate() == 16000 {
        wave.samples().to_vec()
    } else {
        sherpa_onnx::LinearResampler::create(wave.sample_rate(), 16000)
            .unwrap()
            .resample(wave.samples(), true)
    };
    for _ in 0..2 {
        let t = Instant::now();
        let text = listen::transcribe(&recognizer, &samples);
        println!(
            "STT: \"{text}\" ({:.1}s of audio in {} ms)",
            samples.len() as f32 / 16000.0,
            t.elapsed().as_millis()
        );
    }

    let t = Instant::now();
    let voice = speak::Voice::load(cfg)?;
    println!("TTS loaded in {} ms, sample rate {}", t.elapsed().as_millis(), voice.sample_rate);
    let texts = [
        "Hey! Ready to go mining?",
        "Watch out, there's a creeper right behind you, run!",
        "Yeah, those two zombies are getting pretty annoying. Want me to distract one so you can sneak past?",
    ];
    for (i, text) in texts.iter().enumerate() {
        let t = Instant::now();
        let (mut first, mut chunks, mut pcm) = (None, 0, Vec::new());
        voice.stream(text, "", |s| {
            first.get_or_insert(t.elapsed().as_millis());
            chunks += 1;
            pcm.extend_from_slice(s);
            true
        })?;
        let audio_secs = pcm.len() as f32 / voice.sample_rate as f32;
        println!(
            "TTS: \"{text}\" first audio {} ms, {chunks} chunks, {audio_secs:.1}s audio in {} ms",
            first.unwrap_or(0),
            t.elapsed().as_millis()
        );
        let path = std::env::temp_dir().join(format!("mate-tts-test-{i}.wav"));
        ptts::wav::write_wav_file(&path, &pcm, voice.sample_rate)?;
        println!("  saved {}", path.display());
    }
    Ok(())
}
