use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone)]
pub struct Config {
    pub mate: MateConfig,
    pub mic: MicConfig,
    pub kev: KevConfig,
    pub llm: LlmConfig,
    pub models: ModelsConfig,
    pub game: GameConfig,
    /// Folder the config file lives in; relative paths are resolved against it.
    #[serde(skip)]
    pub base_dir: PathBuf,
}

#[derive(Deserialize, Clone)]
pub struct MateConfig {
    pub name: String,
    /// A built-in Pocket TTS voice, used when `voice_server` isn't set.
    pub voice: String,
    /// A cloned voice instead: the URL of a Breeze TTS 2 server (start-breeze.ps1)...
    pub voice_server: Option<String>,
    /// ...and the WAV it clones; its transcript is the .txt beside it.
    pub voice_reference: Option<PathBuf>,
    pub voice_temperature: f32,
    pub voice_seed: u64,
}

#[derive(Deserialize, Clone)]
pub struct MicConfig {
    /// Talking over the mate cuts it off. Takes precedence over `mute_while_speaking`.
    #[serde(default = "yes")]
    pub barge_in: bool,
    pub mute_while_speaking: bool,
    /// Save every recognised clip to `recordings/` for debugging.
    #[serde(default)]
    pub save_recordings: bool,
    pub vad_threshold: f32,
    pub end_of_speech_silence: f32,
}

#[derive(Deserialize, Clone)]
pub struct KevConfig {
    pub url: String,
}

#[derive(Deserialize, Clone)]
pub struct LlmConfig {
    pub url: String,
    pub temperature: f32,
}

#[derive(Deserialize, Clone)]
pub struct ModelsConfig {
    pub silero_vad: PathBuf,
    pub parakeet_dir: PathBuf,
}

#[derive(Deserialize, Clone)]
pub struct GameConfig {
    pub port: u16,
}

impl Config {
    /// Loads the TOML file and makes every model path absolute relative to it.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text).context("parsing config")?;
        let base = path.parent().unwrap_or(Path::new("."));
        cfg.base_dir = base.to_path_buf();
        for p in [&mut cfg.models.silero_vad, &mut cfg.models.parakeet_dir].into_iter().chain(cfg.mate.voice_reference.as_mut()) {
            if p.is_relative() {
                *p = base.join(&*p);
            }
        }
        Ok(cfg)
    }
}

fn yes() -> bool {
    true
}

/// sherpa-onnx wants paths as `String`s.
pub fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}
