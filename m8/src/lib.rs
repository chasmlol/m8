//! m8: a voice brain for the characters of any game. The player talks out loud; m8 hears it,
//! the one they talk to answers in a voice of their own, and the game plays it.
//!
//! mic -> Silero VAD -> speech to text -> the speaker's feed -> Kev (decide, optional) -> LLM
//! (words) -> TTS (a cloned voice) -> the game, with the mouth's shapes for what's said (lips.rs)
//!
//! Each model is a server m8 calls (its config); a game is whatever implements [`game::Game`]:
//! Halo's mod over a socket ([`link::GameLink`], the `m8` binary), or a VR runtime's plugin that
//! runs m8 for every character in its game.

pub mod brain;
pub mod config;
pub mod game;
pub mod link;
pub mod lips;
pub mod listen;
pub mod llm;
pub mod picture;
pub mod site;
pub mod speak;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Shared state about the voice: when its queued speech will finish playing, and a generation
/// counter that the player bumps by talking over it. Everything produced for an older generation
/// (LLM tokens, TTS audio) is dropped. One voice speaks at a time, whoever's it is.
#[derive(Clone)]
pub struct Speaking {
    until: Arc<Mutex<Instant>>,
    generation: Arc<AtomicU64>,
    /// Lines begun (the LLM writing them) whose voicing hasn't finished: speak.rs counts each
    /// off at its end.
    lines: Arc<AtomicUsize>,
}

impl Default for Speaking {
    fn default() -> Self {
        Self::new()
    }
}

impl Speaking {
    /// Extra time counted as "still speaking" after the audio, covering playback latency and room echo.
    const TAIL: Duration = Duration::from_millis(500);

    pub fn new() -> Self {
        Speaking {
            until: Arc::new(Mutex::new(Instant::now())),
            generation: Arc::new(AtomicU64::new(0)),
            lines: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// A line begun: until its end, someone's talking.
    pub fn begin_line(&self) {
        self.lines.fetch_add(1, Ordering::SeqCst);
    }

    /// A line's voicing finished (or was dropped).
    pub fn end_line(&self) {
        let _ = self.lines.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
    }

    /// When the voice stops (or stopped): none while a line is still being written or voiced.
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

    /// The player talked over the speaker: abandon everything they were about to say.
    pub fn interrupt(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        *self.until.lock().unwrap() = Instant::now() - Self::TAIL;
    }
}
