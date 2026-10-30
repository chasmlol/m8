use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone)]
pub struct Config {
    pub mate: MateConfig,
    pub mic: MicConfig,
    pub stt: SttConfig,
    #[serde(default)]
    pub kev: KevConfig,
    pub llm: LlmConfig,
    pub models: ModelsConfig,
    #[serde(default)]
    pub game: GameConfig,
    /// Folder the config file lives in; relative paths are resolved against it. The memory
    /// (`saves.json`), `player_name.txt` and `logs/` are kept here too.
    #[serde(skip)]
    pub base_dir: PathBuf,
}

/// The speaker the game doesn't describe itself: a game with one companion (Halo's Kevin) leaves
/// who they are to the persona folder. A game that describes each of its speakers in its state
/// (`name`, `persona`, `voice`) needs none of this but the voice server.
#[derive(Deserialize, Clone)]
pub struct MateConfig {
    #[serde(default = "mate")]
    pub name: String,
    /// The persona's folder: system_prompt.txt (who the mate is), combat_prompt.txt and
    /// radio_prompt.txt (added while the game says so; either may be missing) and levels/ (each
    /// level's story, by the name the game gives it). The config's folder if unset.
    #[serde(default)]
    pub persona: Option<PathBuf>,
    /// The voice: a server with Breeze TTS 2's API (speak.rs; start-breeze.ps1)...
    pub voice_server: String,
    /// ...and the WAV it clones (its transcript is the .txt beside it), for a speaker that brings
    /// no voice of its own.
    #[serde(default)]
    pub voice_reference: Option<PathBuf>,
    /// Its noise seed, so every line comes out in the same voice.
    #[serde(default = "seed")]
    pub voice_seed: u64,
}

#[derive(Deserialize, Clone)]
pub struct MicConfig {
    /// Talking over the mate cuts it off. Takes precedence over `mute_while_speaking`.
    #[serde(default = "yes")]
    pub barge_in: bool,
    #[serde(default = "yes")]
    pub mute_while_speaking: bool,
    /// Save every recognised clip to `recordings/` for debugging.
    #[serde(default)]
    pub save_recordings: bool,
    #[serde(default = "vad_threshold")]
    pub vad_threshold: f32,
    #[serde(default = "end_of_speech_silence")]
    pub end_of_speech_silence: f32,
    /// The microphone: the first input device whose name has this in it (case ignored), else the
    /// system's default. A headset's own mic (Virtual Desktop's: "Virtual Desktop") rather than
    /// whatever the desktop uses.
    #[serde(default)]
    pub device: Option<String>,
}

/// Speech to text: an OpenAI-style /v1/audio/transcriptions (listen.rs).
#[derive(Deserialize, Clone)]
pub struct SttConfig {
    pub url: String,
    /// The `model` it's sent, for servers that want one.
    #[serde(default)]
    pub model: Option<String>,
}

/// Kev, the decision model: which of the game's actions the speaker takes, and whether something
/// that happened in the game makes them speak. Off, every line the player says to a speaker is
/// answered and nothing else starts one.
#[derive(Deserialize, Clone, Default)]
pub struct KevConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub url: String,
}

/// The LLM: an OpenAI-style /v1/chat/completions, local (llama.cpp's server) or hosted (OpenAI,
/// DeepSeek, OpenRouter, anything that speaks the same API).
#[derive(Deserialize, Clone)]
pub struct LlmConfig {
    pub url: String,
    /// The model it's asked for (a hosted API needs one; a local server serves what it loaded).
    #[serde(default)]
    pub model: Option<String>,
    /// The environment variable holding the API's key, for a hosted API ("DEEPSEEK_API_KEY").
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Or the key itself, in a config kept out of every repository.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "temperature")]
    pub temperature: f32,
    /// The longest answer, in tokens (people answer in a line or two).
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Anything else the API takes, put in every request as it is: DeepSeek's
    /// `thinking = { type = "disabled" }`, say.
    #[serde(default)]
    pub extra: Map<String, Value>,
}

#[derive(Deserialize, Clone)]
pub struct ModelsConfig {
    /// Silero VAD, which finds where speech starts and ends in the mic's audio.
    pub silero_vad: PathBuf,
}

#[derive(Deserialize, Clone)]
pub struct GameConfig {
    #[serde(default = "port")]
    pub port: u16,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig { port: port() }
    }
}

impl Config {
    /// Loads the TOML file and makes every model path absolute relative to it.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text).context("parsing config")?;
        let base = path.parent().unwrap_or(Path::new("."));
        cfg.base_dir = base.to_path_buf();
        cfg.resolve();
        Ok(cfg)
    }

    /// Makes every relative path absolute, against `base_dir`.
    pub fn resolve(&mut self) {
        let base = self.base_dir.clone();
        let paths = [&mut self.models.silero_vad]
            .into_iter()
            .chain(self.mate.voice_reference.as_mut())
            .chain(self.mate.persona.as_mut());
        for p in paths {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
    }

    /// The persona's folder (`persona`), else the config's.
    pub fn persona_dir(&self) -> PathBuf {
        self.mate.persona.clone().unwrap_or_else(|| self.base_dir.clone())
    }
}

fn yes() -> bool {
    true
}

fn mate() -> String {
    "the mate".to_string()
}

fn seed() -> u64 {
    42
}

fn vad_threshold() -> f32 {
    0.5
}

fn end_of_speech_silence() -> f32 {
    0.1
}

fn temperature() -> f32 {
    1.0
}

fn port() -> u16 {
    47800
}

/// sherpa-onnx wants paths as `String`s.
pub fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}
