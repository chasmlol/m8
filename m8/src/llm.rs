//! The LLM's endpoint: an OpenAI-style /v1/chat/completions, local or hosted. Every request goes
//! through here, so a hosted API gets its model, its key and its own extras on each one, over one
//! connection kept open between them (`keep_warm`). OpenRouter gets its own too: who's calling
//! (its attribution headers), which hosts serve the model (its provider routing), and its errors
//! said in plain words, a bad key or model found as m8 starts.

use crate::config::{LlmConfig, Sort};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A hosted API drops a connection left idle for a minute or so, and a new one costs a TLS
/// handshake (0.3 to 0.5 s) on the next line: one is asked something that runs no model (its
/// `/models`; OpenRouter's is most of a megabyte, so its `/key`) when nothing's gone to it this
/// long.
const KEEP_WARM: Duration = Duration::from_secs(25);

#[derive(Clone)]
pub struct Llm {
    url: String,
    model: Option<String>,
    key: Option<String>,
    /// The environment variable the key comes from, for the errors.
    key_env: Option<String>,
    max_tokens: Option<u32>,
    /// What's put in every request: the config's extras, and its reasoning and OpenRouter's
    /// routing as the API takes them.
    extra: serde_json::Map<String, Value>,
    /// OpenRouter: its headers on every request.
    openrouter: bool,
    headers: Vec<(&'static str, String)>,
    pub temperature: f32,
    agent: ureq::Agent,
    /// When anything last went to it.
    last: Arc<Mutex<Instant>>,
}

impl Llm {
    /// The key is the config's own, else its environment variable's (none for a local server).
    pub fn new(cfg: &LlmConfig) -> Llm {
        let url = cfg.url().to_string();
        let key_env = cfg.api_key_env().map(str::to_string);
        let key = cfg
            .api_key
            .clone()
            .filter(|key| !key.trim().is_empty())
            .or_else(|| key_env.as_deref().and_then(|name| std::env::var(name).ok()))
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty());
        if url.is_empty() {
            eprintln!("[llm] no LLM: set [llm] preset (\"openrouter\", \"openai\", \"local\") or url");
        } else if key.is_none() {
            if let Some(name) = &key_env {
                eprintln!("[llm] no key in the environment variable {name}: the LLM at {url} will refuse (set it, then start again)");
            }
        }
        let openrouter = cfg.is_openrouter();
        let mut extra = cfg.extra.clone();
        let mut headers = Vec::new();
        if openrouter {
            let or = &cfg.openrouter;
            if let Some(url) = or.app_url.as_deref().filter(|u| !u.is_empty()) {
                headers.push(("HTTP-Referer", url.to_string()));
            }
            if let Some(title) = or.app_title.as_deref().filter(|t| !t.is_empty()) {
                // (its newer name, and the one older servers know)
                headers.push(("X-OpenRouter-Title", title.to_string()));
                headers.push(("X-Title", title.to_string()));
            }
            let mut provider = serde_json::Map::new();
            if let Some(sort) = or.sort {
                provider.insert("sort".into(), json!(match sort {
                    Sort::Latency => "latency",
                    Sort::Throughput => "throughput",
                    Sort::Price => "price",
                }));
            }
            for (name, hosts) in [("order", &or.order), ("only", &or.only), ("ignore", &or.ignore)] {
                if !hosts.is_empty() {
                    provider.insert(name.into(), json!(hosts));
                }
            }
            if let Some(allow) = or.allow_fallbacks {
                provider.insert("allow_fallbacks".into(), json!(allow));
            }
            if or.deny_data_collection {
                provider.insert("data_collection".into(), json!("deny"));
            }
            if or.zdr {
                provider.insert("zdr".into(), json!(true));
            }
            if !provider.is_empty() {
                extra.entry("provider").or_insert(Value::Object(provider));
            }
            if let Some(effort) = cfg.reasoning {
                extra.entry("reasoning").or_insert(json!({"effort": effort.name()}));
            }
        } else if let Some(effort) = cfg.reasoning {
            extra.entry("reasoning_effort").or_insert(json!(effort.name()));
        }
        Llm {
            url,
            model: cfg.model.clone().filter(|m| !m.trim().is_empty()),
            key,
            key_env,
            max_tokens: cfg.max_tokens,
            extra,
            openrouter,
            headers,
            temperature: cfg.temperature,
            agent: ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build(),
            last: Arc::new(Mutex::new(Instant::now() - KEEP_WARM)),
        }
    }

    fn local(&self) -> bool {
        ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|l| self.url.starts_with(l))
    }

    /// Its API's base (its URL without `/chat/completions`).
    fn base(&self) -> Option<String> {
        self.url.rsplit_once("/chat/completions").map(|(base, _)| base.to_string())
    }

    /// Keeps the connection to a hosted API open from now on, so no line waits for a new one: on a
    /// thread of its own, its `/models` (OpenRouter's `/key`) whenever nothing else has gone to it
    /// for a while (the first at once). A server on this machine needs none of it. OpenRouter's
    /// key and model are checked first, so a bad one is said at the start, not at the first line.
    pub fn keep_warm(&self) {
        let Some(base) = self.base().filter(|_| !self.local()) else { return };
        let llm = self.clone();
        std::thread::spawn(move || {
            if llm.openrouter {
                llm.check_openrouter(&base);
            }
            let ping = if llm.openrouter { format!("{base}/key") } else { format!("{base}/models") };
            loop {
                let idle = llm.last.lock().unwrap().elapsed();
                if idle >= KEEP_WARM {
                    *llm.last.lock().unwrap() = Instant::now();
                    let mut get = llm.agent.get(&ping);
                    if let Some(key) = &llm.key {
                        get = get.set("Authorization", &format!("Bearer {key}"));
                    }
                    match get.call() {
                        Ok(response) => {
                            let _ = response.into_string();
                        }
                        Err(ureq::Error::Status(code, response)) => {
                            eprintln!("[llm] keeping the connection to {base} open: {}", llm.refused(code, response))
                        }
                        Err(e) => eprintln!("[llm] keeping the connection to {base} open: {e}"),
                    }
                }
                std::thread::sleep(KEEP_WARM.saturating_sub(idle).max(Duration::from_secs(1)));
            }
        });
    }

    /// OpenRouter's say on the key (it takes it; credit left) and the model (there's one by that
    /// name), said plainly if not.
    fn check_openrouter(&self, base: &str) {
        *self.last.lock().unwrap() = Instant::now();
        let mut get = self.agent.get(&format!("{base}/key"));
        if let Some(key) = &self.key {
            get = get.set("Authorization", &format!("Bearer {key}"));
        }
        match get.call() {
            Ok(response) => {
                let info: Value = response.into_json().unwrap_or_default();
                let data = &info["data"];
                match data["limit_remaining"].as_f64() {
                    Some(left) if left <= 0.0 => eprintln!("[llm] OpenRouter: the key has no credit left (its limit is used up): openrouter.ai/settings/keys"),
                    _ => println!("[llm] OpenRouter takes the key{}", data["label"].as_str().map(|l| format!(" ({l})")).unwrap_or_default()),
                }
            }
            Err(ureq::Error::Status(code, response)) => eprintln!("[llm] {}", self.refused(code, response)),
            Err(e) => eprintln!("[llm] OpenRouter can't be reached: {e}"),
        }
        let Some(model) = &self.model else {
            eprintln!("[llm] OpenRouter needs a model: set [llm] model to one from openrouter.ai/models (its id, as \"deepseek/deepseek-v4.1-flash\")");
            return;
        };
        // (its endpoints: a few KB, where the list of every model is most of a megabyte)
        let id = model.split(':').next().unwrap_or(model);
        match self.agent.get(&format!("{base}/models/{id}/endpoints")).call() {
            Ok(response) => {
                let info: Value = response.into_json().unwrap_or_default();
                let hosts = info["data"]["endpoints"].as_array().map_or(0, Vec::len);
                println!("[llm] OpenRouter: {model}, served by {hosts} host{}", if hosts == 1 { "" } else { "s" });
            }
            Err(ureq::Error::Status(404, _)) => eprintln!(
                "[llm] OpenRouter has no model \"{model}\": copy the id from its page on openrouter.ai/models (as \"deepseek/deepseek-v4.1-flash\")"
            ),
            Err(e) => eprintln!("[llm] OpenRouter: couldn't look up the model {model}: {e}"),
        }
    }

    /// An error the API answered with, in plain words where it's a common one, with what it said.
    fn refused(&self, code: u16, response: ureq::Response) -> String {
        let body = response.into_string().unwrap_or_default();
        let said = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| body.trim().chars().take(400).collect());
        let model = self.model.as_deref().unwrap_or("(none set)");
        let key = match (&self.key_env, &self.key) {
            (Some(name), None) => format!("there's no key in the environment variable {name}"),
            (Some(name), Some(_)) => format!("the key in the environment variable {name} isn't taken"),
            (None, Some(_)) => "the key in [llm] api_key isn't taken".to_string(),
            (None, None) => "it needs a key: set [llm] api_key_env to the environment variable holding it".to_string(),
        };
        let who = if self.openrouter { "OpenRouter".to_string() } else { format!("the LLM at {}", self.url) };
        let plainly = match code {
            401 => format!("{who} refused: {key}"),
            402 if self.openrouter => format!("{who}: out of credit (openrouter.ai/settings/credits), or the key's own limit is used up"),
            402 => format!("{who}: out of credit"),
            403 => format!("{who} won't take this request (a moderation flag or the key's permissions)"),
            404 => format!("{who} has no model \"{model}\" (or no such API at that URL)"),
            408 | 504 => format!("{who}: the model took too long"),
            429 => format!("{who}: too many requests just now (rate limited; a free model's daily limit, perhaps)"),
            502 => format!("{who}: the model \"{model}\" is down or answered badly"),
            503 if self.openrouter => format!("{who}: no host serves \"{model}\" as [llm.openrouter] asks (its only, ignore, zdr...)"),
            _ => format!("{who} answered {code}"),
        };
        format!("{plainly} ({code}: {said})")
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
        if self.url.is_empty() {
            return Err(anyhow!("no LLM: set [llm] preset or url"));
        }
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
        for (name, value) in &self.headers {
            post = post.set(name, value);
        }
        post.send_json(&request).map_err(|e| match e {
            ureq::Error::Status(code, response) => anyhow!("{}", self.refused(code, response)),
            other => anyhow!("the LLM at {}: {other}", self.url),
        })
    }
}

/// An error the API sent in its answer rather than as its status (OpenRouter's, once a streamed
/// answer has begun: `{"error": {"code", "message"}}` with the finish reason "error"), in words.
pub fn error_in(chunk: &Value) -> Option<String> {
    let error = chunk.get("error").filter(|e| !e.is_null())?;
    let message = error["message"].as_str().map(str::to_string).unwrap_or_else(|| error.to_string());
    Some(match error["code"].as_u64() {
        Some(code) => format!("the LLM stopped ({code}: {message})"),
        None => format!("the LLM stopped ({message})"),
    })
}
