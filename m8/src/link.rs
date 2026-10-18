//! TCP link to the game (a mod, or a port of it). We listen on localhost; the game connects and reconnects.
//!
//! Frames are `[u8 kind][u32 big-endian length][payload]`. Kind 1 is a JSON object,
//! kind 2 is raw mono 16-bit little-endian PCM of the mate's voice. Kind 3, from the game, is
//! dialogue it played for the mate to hear: `[u16 big-endian length][JSON: speaker, radio,
//! react, sample_rate][mono 16-bit little-endian PCM]`. Kind 4, from the game, the player's view
//! the mate asked for (`view`): `[u16 big-endian length][JSON: width, height, spots][RGB rows]`.

use serde_json::{json, Value};
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::Sender;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const KIND_JSON: u8 = 1;
const KIND_AUDIO: u8 = 2;
const KIND_HEAR: u8 = 3;
const KIND_VIEW: u8 = 4;

/// The player's view, as the game sent it: RGB rows from the top, and the spots marked on it
/// (`id`, `x` and `y` from 0 to 1 across and down, `metres` from the player).
pub struct View {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub spots: Vec<Value>,
}

/// Dialogue the game played: what it says of it, and the samples (-1 to 1).
pub type Heard = (Value, Vec<f32>);

#[derive(Clone)]
pub struct GameLink {
    writer: Arc<Mutex<Option<TcpStream>>>,
    /// The most recent `state` message from the mod.
    pub state: Arc<Mutex<Value>>,
    /// The player muted the mic to the mate (a key in the game: `mic` messages).
    pub mic_muted: Arc<AtomicBool>,
    /// After the mate's voice stops, how long (ms) before the game's events can have him talk
    /// again: the game's setting (`settings` messages).
    speech_cooldown_ms: Arc<AtomicU64>,
    /// The view last asked for, once it comes.
    view: Arc<(Mutex<Option<View>>, Condvar)>,
    hello: Value,
}

impl GameLink {
    /// Starts accepting connections in the background. The game's events, saves and loads go to
    /// `events`, dialogue it played to `hearing`.
    pub fn start(port: u16, hello: Value, events: Sender<Value>, hearing: Sender<Heard>) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let link = GameLink {
            writer: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(Value::Null)),
            mic_muted: Arc::new(AtomicBool::new(false)),
            speech_cooldown_ms: Arc::new(AtomicU64::new(5000)),
            view: Arc::new((Mutex::new(None), Condvar::new())),
            hello,
        };

        let accept_link = link.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let _ = stream.set_nodelay(true);
                let Ok(reader) = stream.try_clone() else { continue };
                println!("[game] game connected");
                *accept_link.writer.lock().unwrap() = Some(stream);
                accept_link.send_json(&accept_link.hello);

                let read_link = accept_link.clone();
                let (events, hearing) = (events.clone(), hearing.clone());
                std::thread::spawn(move || read_link.read_loop(reader, events, hearing));
            }
        });
        Ok(link)
    }

    pub fn speech_cooldown(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.speech_cooldown_ms.load(Ordering::Relaxed))
    }

    /// Asks the game for the player's view, and waits for it (None: no game, or it didn't come).
    pub fn view(&self, wait: Duration) -> Option<View> {
        let (slot, came) = &*self.view;
        let mut view = slot.lock().unwrap();
        *view = None;
        self.send_json(&json!({"type": "view"}));
        let (mut view, _) = came.wait_timeout_while(view, wait, |view| view.is_none()).unwrap();
        view.take()
    }

    pub fn send_json(&self, value: &Value) {
        self.send_frame(KIND_JSON, value.to_string().as_bytes());
    }

    pub fn send_audio(&self, pcm: &[u8]) {
        self.send_frame(KIND_AUDIO, pcm);
    }

    pub fn say(&self, text: &str) {
        self.send_json(&json!({"type": "say", "text": text}));
    }

    fn send_frame(&self, kind: u8, payload: &[u8]) {
        let mut guard = self.writer.lock().unwrap();
        let Some(stream) = guard.as_mut() else { return };
        let mut frame = Vec::with_capacity(5 + payload.len());
        frame.push(kind);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        if stream.write_all(&frame).is_err() {
            println!("[game] game disconnected");
            *guard = None;
        }
    }

    fn read_loop(&self, stream: TcpStream, events: Sender<Value>, hearing: Sender<Heard>) {
        let mut reader = BufReader::new(stream);
        loop {
            let mut header = [0u8; 5];
            if reader.read_exact(&mut header).is_err() {
                break;
            }
            let len = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                break;
            }
            if header[0] == KIND_HEAR && payload.len() >= 2 {
                let length = u16::from_be_bytes([payload[0], payload[1]]) as usize;
                let Some(about) = payload.get(2..2 + length).and_then(|j| serde_json::from_slice::<Value>(j).ok()) else {
                    continue;
                };
                let samples = payload[2 + length..]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                let _ = hearing.send((about, samples));
                continue;
            }
            if header[0] == KIND_VIEW && payload.len() >= 2 {
                let length = u16::from_be_bytes([payload[0], payload[1]]) as usize;
                let Some(about) = payload.get(2..2 + length).and_then(|j| serde_json::from_slice::<Value>(j).ok()) else {
                    continue;
                };
                let (width, height) = (about["width"].as_u64().unwrap_or(0) as usize, about["height"].as_u64().unwrap_or(0) as usize);
                let rgb = payload[2 + length..].to_vec();
                if width * height * 3 == rgb.len() {
                    let spots = about["spots"].as_array().cloned().unwrap_or_default();
                    let (slot, came) = &*self.view;
                    *slot.lock().unwrap() = Some(View { width, height, rgb, spots });
                    came.notify_all();
                }
                continue;
            }
            if header[0] != KIND_JSON {
                continue;
            }
            let Ok(msg) = serde_json::from_slice::<Value>(&payload) else { continue };
            match msg["type"].as_str() {
                Some("state") => *self.state.lock().unwrap() = msg,
                Some("settings") => {
                    if let Some(seconds) = msg["speech_cooldown"].as_f64() {
                        self.speech_cooldown_ms.store((seconds.max(0.0) * 1000.0) as u64, Ordering::Relaxed);
                        println!("[game] speech cooldown {seconds} s");
                    }
                }
                Some("mic") => {
                    let muted = msg["muted"].as_bool().unwrap_or(false);
                    if self.mic_muted.swap(muted, Ordering::Relaxed) != muted {
                        println!("[mic] {}", if muted { "muted in the game" } else { "unmuted in the game" });
                    }
                }
                Some("event" | "saved" | "loaded" | "direct") => {
                    let _ = events.send(msg);
                }
                _ => {}
            }
        }
    }
}
