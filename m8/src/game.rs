//! What the brain needs of a game, whichever game it is and however it's reached: Halo's mod over
//! a socket (link.rs), or a game inside a VR runtime that runs m8 as its plugin.
//!
//! The game has speakers, each its own id: a game with one companion has only `MATE`. For each it
//! gives a state (JSON, as link.rs's `state` message: `state` in words, `actions` for Kev, and the
//! speaker's own `name`, `persona` and `voice` where the game describes them). The brain answers
//! with what the speaker does and says, sentence by sentence with its voice.

use crate::link::View;
use serde_json::Value;
use std::time::Duration;

/// The speaker of a game that has only one, and the one the player talks to when the game doesn't
/// say who.
pub const MATE: &str = "mate";

pub trait Game: Send + Sync {
    /// The game's latest state for a speaker (null before it sent any).
    fn state(&self, speaker: &str) -> Value;

    /// Kev picked what the speaker does: `{"type": "action", "action": ..., "option": ...}`.
    fn act(&self, speaker: &str, action: &Value);

    /// What the player said to the speaker, as it was heard.
    fn heard(&self, speaker: &str, text: &str);

    /// The player started talking (the mic heard speech), before a word of it is known: the one
    /// they talk to can turn to listen.
    fn listening(&self) {}

    /// A sentence of the speaker's line begins: what's said, and how ("[shouting]"'s words).
    fn sentence(&self, speaker: &str, text: &str, how: &str);

    /// The sentence's voice as it's made: mono samples (-1 to 1) at `rate`.
    fn audio(&self, speaker: &str, samples: &[f32], rate: u32);

    /// The sentence's voice is all there.
    fn sentence_end(&self, _speaker: &str) {}

    /// The speaker's whole line is voiced.
    fn line_end(&self, speaker: &str);

    /// Stop what's playing: the player talked over the speaker, or a save was loaded.
    fn stop_audio(&self);

    /// The player muted the mic in the game.
    fn mic_muted(&self) -> bool {
        false
    }

    /// The game is still saying a line, after m8 has sent all its voice (it plays it later than
    /// it was made: the game waited on the voice server).
    fn still_speaking(&self) -> bool {
        false
    }

    /// After a speaker's voice stops, how long before the game's events can have them talk again.
    fn speech_cooldown(&self) -> Duration {
        Duration::from_secs(5)
    }

    /// The player's view, when an action needs it (None: the game has none to give).
    fn view(&self, _wait: Duration) -> Option<View> {
        None
    }

    /// Who the player is talking to (None: `MATE`).
    fn addressed(&self) -> Option<String> {
        None
    }

    /// The state the brain page shows, and the page's stand-in for one (with no game connected).
    fn shown_state(&self) -> Value {
        self.state(MATE)
    }

    fn set_shown_state(&self, _state: Value) {}
}
