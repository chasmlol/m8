//! Open-mic speech recognition: microphone -> Silero VAD -> Parakeet TDT.

use crate::config::{path_str, Config};
use crate::link::GameLink;
use serde_json::json;
use anyhow::{anyhow, bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use sherpa_onnx::{
    LinearResampler, OfflineRecognizer, OfflineRecognizerConfig, VadModelConfig,
    VoiceActivityDetector,
};
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::time::Instant;

use crate::Speaking;

const SAMPLE_RATE: i32 = 16000;
/// Silero works on fixed 512-sample windows at 16 kHz.
const VAD_WINDOW: usize = 512;
/// Mic audio kept around each piece of speech the VAD finds, before and after it: its cut
/// clips a short word's start and end, which the recognizer then can't make out.
const MARGIN_BEFORE: usize = SAMPLE_RATE as usize / 5;
const MARGIN_AFTER: usize = SAMPLE_RATE as usize / 10;
/// The mic audio kept for that (more than the longest piece, 20 s).
const KEPT: usize = SAMPLE_RATE as usize * 25;

pub fn create_recognizer(cfg: &Config) -> Result<OfflineRecognizer> {
    let dir = &cfg.models.parakeet_dir;
    let mut rc = OfflineRecognizerConfig::default();
    rc.model_config.transducer.encoder = Some(path_str(&dir.join("encoder.int8.onnx")));
    rc.model_config.transducer.decoder = Some(path_str(&dir.join("decoder.int8.onnx")));
    rc.model_config.transducer.joiner = Some(path_str(&dir.join("joiner.int8.onnx")));
    rc.model_config.tokens = Some(path_str(&dir.join("tokens.txt")));
    rc.model_config.model_type = Some("nemo_transducer".to_string());
    rc.model_config.num_threads = 4;
    OfflineRecognizer::create(&rc).ok_or_else(|| anyhow!("failed to load Parakeet"))
}

pub fn transcribe(recognizer: &OfflineRecognizer, samples: &[f32]) -> String {
    let stream = recognizer.create_stream();
    stream.accept_waveform(SAMPLE_RATE, samples);
    recognizer.decode(&stream);
    stream.get_result().map(|r| r.text.trim().to_string()).unwrap_or_default()
}

fn create_vad(cfg: &Config) -> Result<VoiceActivityDetector> {
    let mut vc = VadModelConfig::default();
    vc.silero_vad.model = Some(path_str(&cfg.models.silero_vad));
    vc.silero_vad.threshold = cfg.mic.vad_threshold;
    vc.silero_vad.min_silence_duration = cfg.mic.end_of_speech_silence;
    // (none: a quick word is speech too)
    vc.silero_vad.min_speech_duration = 0.0;
    vc.silero_vad.max_speech_duration = 20.0;
    vc.silero_vad.window_size = VAD_WINDOW as i32;
    vc.sample_rate = SAMPLE_RATE;
    VoiceActivityDetector::create(&vc, 60.0).ok_or_else(|| anyhow!("failed to load Silero VAD"))
}

/// Turns raw audio at the input's sample rate into finished utterances.
pub struct Listener {
    cfg: Config,
    recognizer: OfflineRecognizer,
    vad: VoiceActivityDetector,
    resampler: Option<LinearResampler>,
    pending: Vec<f32>,
    was_muted: bool,
    /// How much loud audio arrived while muted, to tell the user they were talked over.
    loud_while_muted: usize,
    recordings: Option<PathBuf>,
    segment_index: usize,
    /// The mic audio the VAD has had lately, from sample `kept_from` on (counting every sample
    /// it has had), and where its own count started (it starts again when reset).
    kept: std::collections::VecDeque<f32>,
    kept_from: usize,
    accepted: usize,
    vad_origin: usize,
}

impl Listener {
    pub fn new(cfg: Config, recognizer: OfflineRecognizer, input_rate: i32) -> Result<Self> {
        let recordings = cfg.mic.save_recordings.then(|| cfg.base_dir.join("recordings"));
        if let Some(dir) = &recordings {
            std::fs::create_dir_all(dir)?;
            println!("[mic] saving what it hears to {}", dir.display());
        }
        Ok(Listener {
            vad: create_vad(&cfg)?,
            resampler: (input_rate != SAMPLE_RATE)
                .then(|| LinearResampler::create(input_rate, SAMPLE_RATE))
                .flatten(),
            cfg,
            recognizer,
            pending: Vec::new(),
            was_muted: false,
            loud_while_muted: 0,
            recordings,
            segment_index: 0,
            kept: std::collections::VecDeque::new(),
            kept_from: 0,
            accepted: 0,
            vad_origin: 0,
        })
    }

    /// Feeds one chunk of input audio; returns the text of any utterances it completed.
    pub fn feed(&mut self, chunk: Vec<f32>, muted: bool) -> Vec<String> {
        let samples = match &self.resampler {
            Some(r) => r.resample(&chunk, false),
            None => chunk,
        };

        // Half duplex: drop mic audio while the mate is talking (plus a short tail).
        if muted {
            if !self.was_muted {
                self.vad.reset();
                self.vad_origin = self.accepted;
                self.pending.clear();
                self.loud_while_muted = 0;
            }
            self.was_muted = true;
            if level_db(&samples).0 > -40.0 {
                self.loud_while_muted += samples.len();
            }
            return Vec::new();
        }
        if self.was_muted && self.loud_while_muted > SAMPLE_RATE as usize * 3 / 10 {
            println!(
                "[mic] ignored ~{:.1}s of sound while {} was talking (with headphones, set mute_while_speaking = false)",
                self.loud_while_muted as f32 / SAMPLE_RATE as f32,
                self.cfg.mate.name
            );
        }
        self.was_muted = false;

        self.pending.extend_from_slice(&samples);
        let mut offset = 0;
        while offset + VAD_WINDOW <= self.pending.len() {
            let window = &self.pending[offset..offset + VAD_WINDOW];
            self.vad.accept_waveform(window);
            self.kept.extend(window);
            self.accepted += VAD_WINDOW;
            offset += VAD_WINDOW;
        }
        self.pending.drain(..offset);
        if self.kept.len() > KEPT {
            let excess = self.kept.len() - KEPT;
            self.kept.drain(..excess);
            self.kept_from += excess;
        }

        let mut heard = Vec::new();
        while let Some(segment) = self.vad.front() {
            self.vad.pop();
            let audio = self.with_margins(segment.start(), segment.samples());
            let audio = audio.as_slice();
            let start = Instant::now();
            let text = transcribe(&self.recognizer, audio);
            let (peak, rms) = level_db(audio);
            println!(
                "[stt] \"{}\" ({:.1}s audio, peak {:.0} dB, rms {:.0} dB, {} ms)",
                text,
                audio.len() as f32 / SAMPLE_RATE as f32,
                peak,
                rms,
                start.elapsed().as_millis()
            );
            if let Some(dir) = &self.recordings {
                self.segment_index += 1;
                let path = dir.join(format!("{:03}.wav", self.segment_index));
                if let Err(e) = ptts::wav::write_wav_file(&path, audio, SAMPLE_RATE as u32) {
                    eprintln!("[mic] couldn't save {}: {e}", path.display());
                }
            }
            if !text.is_empty() {
                heard.push(text);
            }
        }
        heard
    }
}

impl Listener {
    /// A piece of speech the VAD found (its start in its own count, and its audio), with the
    /// mic audio around it: MARGIN_BEFORE before, MARGIN_AFTER after, as far as there is.
    fn with_margins(&self, start: i32, samples: &[f32]) -> Vec<f32> {
        let start = self.vad_origin + start.max(0) as usize;
        let from = start.saturating_sub(MARGIN_BEFORE).max(self.kept_from);
        let to = (start + samples.len() + MARGIN_AFTER).min(self.accepted);
        if from >= to || start < self.kept_from {
            return samples.to_vec();
        }
        self.kept.range(from - self.kept_from..to - self.kept_from).copied().collect()
    }

    /// True while the VAD thinks someone is talking (it needs ~0.25 s of speech to switch on).
    pub fn speech_active(&self) -> bool {
        self.vad.detected()
    }
}

/// Runs forever on the calling thread, sending each finished utterance's text to `out`.
///
/// With barge-in on, the mic stays live while the mate talks: as soon as the player starts
/// speaking, the mate is cut off (queued audio in the game, pending TTS and the LLM reply).
pub fn run(cfg: Config, recognizer: OfflineRecognizer, speaking: Speaking, link: GameLink, out: Sender<String>) -> Result<()> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow!("no microphone found"))?;
    let supported = device.default_input_config()?;
    let mic_rate = supported.sample_rate().0 as i32;
    println!(
        "[mic] using '{}' at {} Hz, {} channel(s), {:?}",
        device.name().unwrap_or_default(),
        mic_rate,
        supported.channels(),
        supported.sample_format()
    );

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = build_input_stream(&device, &supported, tx)?;
    stream.play()?;

    let (barge_in, mute_while_speaking) = (cfg.mic.barge_in, cfg.mic.mute_while_speaking);
    let name = cfg.mate.name.clone();
    let mut listener = Listener::new(cfg, recognizer, mic_rate)?;
    for chunk in rx {
        let mate_talking = speaking.is_speaking();
        // (muted in the game: nothing heard, and he isn't cut off; the listener starts over,
        // without counting the sound as talked over him)
        if link.mic_muted.load(std::sync::atomic::Ordering::Relaxed) {
            listener.feed(Vec::new(), true);
            continue;
        }
        let heard = listener.feed(chunk, !barge_in && mute_while_speaking && mate_talking);
        if barge_in && mate_talking && listener.speech_active() {
            println!("[mic] you talked over {name}, cutting them off");
            speaking.interrupt();
            link.send_json(&json!({"type": "stop_audio"}));
        }
        for text in heard {
            if out.send(text).is_err() {
                return Ok(());
            }
        }
    }
    Ok(())
}
/// Peak and RMS level in dBFS.
fn level_db(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (-100.0, -100.0);
    }
    let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let db = |x: f32| 20.0 * x.max(1e-5).log10();
    (db(peak), db(rms))
}

fn build_input_stream(
    device: &cpal::Device,
    supported: &cpal::SupportedStreamConfig,
    tx: Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let config = supported.config();
    let channels = config.channels as usize;
    let err_fn = |err| eprintln!("[mic] stream error: {err:?}");

    fn to_mono<T: Copy>(data: &[T], channels: usize, conv: impl Fn(T) -> f32) -> Vec<f32> {
        data.chunks(channels)
            .map(|frame| frame.iter().map(|&s| conv(s)).sum::<f32>() / channels as f32)
            .collect()
    }

    let stream = match supported.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                let _ = tx.send(to_mono(data, channels, |s| s));
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let _ = tx.send(to_mono(data, channels, |s| s as f32 / i16::MAX as f32));
            },
            err_fn,
            None,
        )?,
        other => bail!("unsupported microphone sample format {other:?}"),
    };
    Ok(stream)
}
