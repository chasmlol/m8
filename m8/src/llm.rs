//! The LLM's endpoint: an OpenAI-style /v1/chat/completions, local or hosted. Every request goes
//! through here, so a hosted API gets its model, its key and its own extras on each one.

use crate::config::LlmConfig;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::time::Duration;

#[derive(Clone)]
pub struct Llm {
    url: String,
    model: Option<String>,
    key: Option<String>,
    max_tokens: Option<u32>,
    extra: serde_json::Map<String, Value>,
    pub temperature: f32,
    agent: ureq::Agent,
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
        }
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
