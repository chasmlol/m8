//! The LLM's endpoint: an OpenAI-style /v1/chat/completions, local or hosted. Every request goes
//! through here, so a hosted API gets its model, its key and its own extras on each one, over one
//! connection kept open between them (`keep_warm`).

use crate::config::LlmConfig;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A hosted API drops a connection left idle for a minute or so, and a new one costs a TLS
/// handshake (0.3 to 0.5 s) on the next line: one is asked something that runs no model (its
/// `/models`) when nothing's gone to it this long.
const KEEP_WARM: Duration = Duration::from_secs(25);

#[derive(Clone)]
pub struct Llm {
    url: String,
    model: Option<String>,
    key: Option<String>,
    max_tokens: Option<u32>,
    extra: serde_json::Map<String, Value>,
    pub temperature: f32,
    agent: ureq::Agent,
    /// When anything last went to it.
    last: Arc<Mutex<Instant>>,
}

impl Llm {
    /// The key is the config's own, else its environment variable's (none for a local server).
    pub fn new(cfg: &LlmConfig) -> Llm {
        let key = cfg
            .api_key
            .clone()
            .filter(|key| !key.trim().is_empty())
            .or_else(|| cfg.api_key_env.as_deref().and_then(|name| std::env::var(name).ok()))
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty());
        if key.is_none() {
            if let Some(name) = &cfg.api_key_env {
                eprintln!("[llm] no key in the environment variable {name}: the LLM at {} will refuse", cfg.url);
            }
        }
        Llm {
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            key,
            max_tokens: cfg.max_tokens,
            extra: cfg.extra.clone(),
            temperature: cfg.temperature,
            agent: ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build(),
            last: Arc::new(Mutex::new(Instant::now() - KEEP_WARM)),
        }
    }

    /// Keeps the connection to a hosted API open from now on, so no line waits for a new one: on a
    /// thread of its own, its `/models` whenever nothing else has gone to it for a while (the
    /// first at once). A server on this machine needs none of it.
    pub fn keep_warm(&self) {
        let local = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|l| self.url.starts_with(l));
        let Some(base) = self.url.rsplit_once("/chat/completions").map(|(base, _)| base.to_string()).filter(|_| !local) else { return };
        let (agent, key, last) = (self.agent.clone(), self.key.clone(), self.last.clone());
        std::thread::spawn(move || loop {
            let idle = last.lock().unwrap().elapsed();
            if idle >= KEEP_WARM {
                *last.lock().unwrap() = Instant::now();
                let mut get = agent.get(&format!("{base}/models"));
                if let Some(key) = &key {
                    get = get.set("Authorization", &format!("Bearer {key}"));
                }
                if let Err(e) = get.call().and_then(|r| Ok(r.into_string()?)) {
                    eprintln!("[llm] keeping the connection to {base} open: {e}");
                }
            }
            std::thread::sleep(KEEP_WARM.saturating_sub(idle).max(Duration::from_secs(1)));
        });
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Sends a request (its `messages`, and whatever else it sets), with the model, the longest
    /// answer and the config's extras where the request doesn't set them. An error the API
    /// answered with comes back with what it said.
    pub fn post(&self, mut request: Value) -> Result<ureq::Response> {
        if let Some(body) = request.as_object_mut() {
            if let Some(model) = &self.model {
                body.entry("model").or_insert_with(|| Value::from(model.clone()));
            }
            if let Some(tokens) = self.max_tokens {
                body.entry("max_tokens").or_insert_with(|| Value::from(tokens));
            }
            for (key, value) in &self.extra {
                body.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        *self.last.lock().unwrap() = Instant::now();
        let mut post = self.agent.post(&self.url);
        if let Some(key) = &self.key {
            post = post.set("Authorization", &format!("Bearer {key}"));
        }
        post.send_json(&request).map_err(|e| match e {
            ureq::Error::Status(code, response) => {
                anyhow!("the LLM at {} answered {code}: {}", self.url, response.into_string().unwrap_or_default())
            }
            other => anyhow!("the LLM at {}: {other}", self.url),
        })
    }
}
