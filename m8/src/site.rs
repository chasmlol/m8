//! A local page (http://127.0.0.1:8090) to watch and poke the brain: the exact prompt the LLM
//! and Kev get, every Kev answer, LLM line and action, a box to talk to the mate, and buttons
//! that send game events, and the system and combat prompts to edit. Its "state" box stands in
//! for the game when none is connected. /history goes through a session entry by entry, each
//! model call with exactly what went in and what came out (logs/<started>.jsonl keeps them).

use crate::brain::Input;
use crate::link::GameLink;
use serde_json::{json, Value};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PORT: u16 = 8090;

/// What the brain did, entry by entry, for the pages; a session's also written to
/// logs/<started>.jsonl (a line an entry), for the history page to read back later.
#[derive(Clone)]
pub struct Trace(Arc<Mutex<Traced>>);

struct Traced {
    /// The LLM's latest prompt.
    prompt: String,
    entries: Vec<Value>,
    /// This session's name (when it started, in seconds since 1970) and its log.
    session: String,
    logs: PathBuf,
    file: Option<File>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

impl Trace {
    pub fn new(logs: &Path) -> Self {
        let session = (now_ms() / 1000).to_string();
        let _ = std::fs::create_dir_all(logs);
        let file = File::create(logs.join(format!("{session}.jsonl")))
            .map_err(|e| eprintln!("[site] no history log in {}: {e}", logs.display()))
            .ok();
        Trace(Arc::new(Mutex::new(Traced { prompt: String::new(), entries: Vec::new(), session, logs: logs.to_path_buf(), file })))
    }

    pub fn set_prompt(&self, prompt: &str) {
        self.0.lock().unwrap().prompt = prompt.to_string();
    }

    pub fn add(&self, kind: &str, data: Value) {
        let mut traced = self.0.lock().unwrap();
        let entry = json!({"id": traced.entries.len(), "t": now_ms(), "kind": kind, "data": data});
        if let Some(file) = &mut traced.file {
            let _ = writeln!(file, "{entry}");
        }
        traced.entries.push(entry);
    }
}

/// The sessions logged, newest first, and the one going on.
fn sessions(trace: &Trace) -> Value {
    let traced = trace.0.lock().unwrap();
    let mut names: Vec<(String, u64)> = std::fs::read_dir(&traced.logs)
        .map(|dir| {
            dir.flatten()
                .filter_map(|file| {
                    let name = file.file_name().to_string_lossy().strip_suffix(".jsonl")?.to_string();
                    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
                    name.chars().all(|c| c.is_ascii_digit()).then_some((name, size))
                })
                .collect()
        })
        .unwrap_or_default();
    names.sort_by(|a, b| b.0.cmp(&a.0));
    json!({
        "live": traced.session,
        "sessions": names.iter().map(|(name, size)| json!({"name": name, "bytes": size})).collect::<Vec<_>>(),
    })
}

/// A logged session's entries (its name: digits only).
fn session(trace: &Trace, name: &str) -> Option<Value> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let path = trace.0.lock().unwrap().logs.join(format!("{name}.jsonl"));
    let text = std::fs::read_to_string(path).ok()?;
    Some(Value::Array(text.lines().filter_map(|line| serde_json::from_str(line).ok()).collect()))
}

/// A prompt kept in a file, so edits on the page last (the system, combat and radio prompts).
#[derive(Clone)]
pub struct PromptFile {
    path: PathBuf,
    text: Arc<Mutex<String>>,
}

impl PromptFile {
    /// (A missing file is an empty prompt: saving one on the page makes it.)
    pub fn load(path: PathBuf) -> anyhow::Result<Self> {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(anyhow::anyhow!("reading {}: {e}", path.display())),
        };
        Ok(PromptFile { path, text: Arc::new(Mutex::new(text)) })
    }

    pub fn get(&self) -> String {
        self.text.lock().unwrap().clone()
    }

    fn set(&self, text: String) -> std::io::Result<()> {
        std::fs::write(&self.path, &text)?;
        *self.text.lock().unwrap() = text;
        Ok(())
    }
}

pub fn run(
    trace: Trace,
    system: PromptFile,
    combat: PromptFile,
    radio: PromptFile,
    link: GameLink,
    inputs: Sender<Input>,
) -> anyhow::Result<()> {
    let server = tiny_http::Server::http(("127.0.0.1", PORT)).map_err(|e| anyhow::anyhow!("{e}"))?;
    for mut request in server.incoming_requests() {
        let mut body = String::new();
        let _ = request.as_reader().read_to_string(&mut body);
        let url = request.url().to_string();
        let reply = match (request.method().as_str(), url.split('?').next().unwrap_or("")) {
            ("GET", "/") => tiny_http::Response::from_string(include_str!("site.html"))
                .with_header("Content-Type: text/html; charset=utf-8".parse::<tiny_http::Header>().unwrap()),
            ("GET", "/history") => tiny_http::Response::from_string(include_str!("history.html"))
                .with_header("Content-Type: text/html; charset=utf-8".parse::<tiny_http::Header>().unwrap()),
            ("GET", "/trace") => {
                let since: usize = url.split("since=").nth(1).and_then(|n| n.parse().ok()).unwrap_or(0);
                let traced = trace.0.lock().unwrap();
                let state = link.state.lock().unwrap().clone();
                let new = traced.entries.get(since..).unwrap_or_default();
                tiny_http::Response::from_string(
                    json!({"prompt": traced.prompt, "entries": new, "next": traced.entries.len(), "state": state,
                           "session": traced.session})
                    .to_string(),
                )
            }
            ("GET", "/sessions") => tiny_http::Response::from_string(sessions(&trace).to_string()),
            // (a picture the LLM saw, logs/views/<name>.png: the mate's eyes)
            ("GET", "/view") => {
                let name = url.split("name=").nth(1).unwrap_or("");
                let path = trace.0.lock().unwrap().logs.join("views").join(format!("{name}.png"));
                match std::fs::read(path).ok().filter(|_| !name.is_empty() && name.chars().all(|c| c.is_ascii_digit())) {
                    Some(png) => tiny_http::Response::from_data(png)
                        .with_header("Content-Type: image/png".parse::<tiny_http::Header>().unwrap()),
                    None => tiny_http::Response::from_data(Vec::new()).with_status_code(404),
                }
            }
            ("GET", "/session") => match session(&trace, url.split("name=").nth(1).unwrap_or("")) {
                Some(entries) => tiny_http::Response::from_string(entries.to_string()),
                None => tiny_http::Response::from_string("no such session").with_status_code(404),
            },
            ("GET", "/system") => tiny_http::Response::from_string(system.get()),
            ("POST", "/system") => match system.set(body) {
                Ok(()) => tiny_http::Response::from_string(""),
                Err(e) => tiny_http::Response::from_string(e.to_string()).with_status_code(500),
            },
            ("GET", "/radio") => tiny_http::Response::from_string(radio.get()),
            ("POST", "/radio") => match radio.set(body) {
                Ok(()) => tiny_http::Response::from_string(""),
                Err(e) => tiny_http::Response::from_string(e.to_string()).with_status_code(500),
            },
            ("GET", "/combat") => tiny_http::Response::from_string(combat.get()),
            ("POST", "/combat") => match combat.set(body) {
                Ok(()) => tiny_http::Response::from_string(""),
                Err(e) => tiny_http::Response::from_string(e.to_string()).with_status_code(500),
            },
            ("POST", "/say") => {
                let _ = inputs.send(Input::Heard(body));
                tiny_http::Response::from_string("")
            }
            ("POST", "/event") => {
                let _ = inputs.send(Input::Event { text: body, react: true });
                tiny_http::Response::from_string("")
            }
            ("POST", "/save") => {
                let _ = inputs.send(Input::Saved(body));
                tiny_http::Response::from_string("")
            }
            ("POST", "/load") => {
                let _ = inputs.send(Input::Loaded(body));
                tiny_http::Response::from_string("")
            }
            ("POST", "/state") => match serde_json::from_str::<Value>(&body) {
                Ok(state) => {
                    *link.state.lock().unwrap() = state;
                    tiny_http::Response::from_string("")
                }
                Err(e) => tiny_http::Response::from_string(e.to_string()).with_status_code(400),
            },
            _ => tiny_http::Response::from_string("").with_status_code(404),
        };
        let _ = request.respond(reply);
    }
    Ok(())
}
