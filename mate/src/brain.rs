//! The mate's brain, the same for every game. Everything goes into one feed: what the player
//! and the mate say, and what happens in the game. The LLM and Kev get the exact same prompt:
//! the system prompt, the level's story so far, what the mate remembers, the feed, and, before
//! its newest lines, what holds right now (the combat or radio prompt while the game says so,
//! and its current state).
//!
//! The player speaks -> the LLM answers -> Kev picks what the mate does.
//! The game reports an event -> Kev picks what the mate does, and whether it speaks.
//!
//! The game decides what the mate can do: its state lists the actions, each with a short
//! description and optionally targets (e.g. which enemy). Kev picks one.
//!
//! Memory: when the feed gets long, the LLM folds its oldest lines into a short summary. The
//! game's saves carry the memory: a save stores it, and loading that save brings it back (dying
//! and going back to a checkpoint forgets what happened since).

use crate::config::Config;
use crate::link::GameLink;
use crate::picture;
use crate::site::{PromptFile, Trace};
use crate::speak::SpeechJob;
use crate::Speaking;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

pub enum Input {
    /// Something the player said (mic, console or the site).
    Heard(String),
    /// Something that happened in the game, in words. `react`: Kev picks what the mate does
    /// about it (and whether it speaks); without, it only goes into the feed (e.g. dialogue in a
    /// cutscene).
    Event { text: String, react: bool },
    /// The game saved (its id for the save), or loaded a save.
    Saved(String),
    Loaded(String),
    /// A summary of the feed's oldest lines is ready.
    Summary { memory: u64, folded: usize, text: String },
}

/// The feed's size, in tokens (counted as 4 characters each): over FEED_TOKENS, at a safe point
/// (the game saved, or it's been quiet), its oldest lines are folded into the summary until it's
/// down to FEED_FOLDED_TO; over FEED_MAX_TOKENS, whenever (the servers take 32k a request).
const FEED_TOKENS: usize = 16_000;
const FEED_FOLDED_TO: usize = 12_000;
const FEED_MAX_TOKENS: usize = 20_000;
/// Nothing from the game or the player, the mate not talking and no fight, this long: it's been
/// quiet, an event of its own.
const QUIET: Duration = Duration::from_secs(45);
const QUIET_EVENT: &str = "It's been quiet for a while.";
/// How many of the feed's newest lines Kev reads (the LLM reads it all): its server reads its
/// whole prompt afresh each call, so the longer it is, the slower every decision.
const KEV_FEED_LINES: usize = 20;
/// The saves remembered (older ones are forgotten).
const SAVES: usize = 50;

/// What the mate remembers: a summary of the past and the feed since.
#[derive(Clone, Default, Serialize, Deserialize)]
struct Memory {
    summary: String,
    feed: Vec<String>,
}

pub struct Brain {
    cfg: Config,
    link: GameLink,
    speech: Sender<SpeechJob>,
    speaking: Speaking,
    agent: ureq::Agent,
    trace: Trace,
    system: PromptFile,
    combat: PromptFile,
    radio: PromptFile,
    /// (for summaries made on another thread)
    inputs: Sender<Input>,
    memory: Memory,
    /// Counts loads, so that a summary of a memory since replaced is dropped.
    memory_generation: u64,
    summarizing: bool,
    saves: Vec<(String, Memory)>,
}

impl Brain {
    pub fn new(
        cfg: Config,
        link: GameLink,
        speech: Sender<SpeechJob>,
        speaking: Speaking,
        trace: Trace,
        system: PromptFile,
        combat: PromptFile,
        radio: PromptFile,
        inputs: Sender<Input>,
    ) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).build();
        let saves = std::fs::read_to_string(Self::saves_path(&cfg))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Brain {
            cfg,
            link,
            speech,
            speaking,
            agent,
            trace,
            system,
            combat,
            radio,
            inputs,
            memory: Memory::default(),
            memory_generation: 0,
            summarizing: false,
            saves,
        }
    }

    pub fn run(mut self, inputs: Receiver<Input>) {
        // (an input taken while gathering events, for next)
        let mut waiting = None;
        let mut quiet_since = Instant::now();
        loop {
            // (a quiet spell: a safe point to fold the feed)
            let mut quiet = false;
            let input = match waiting.take() {
                Some(input) => input,
                None => match inputs.recv_timeout(Duration::from_secs(1)) {
                    Ok(input) => input,
                    Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {
                        let s = self.state();
                        if self.speaking.is_speaking() || s.is_null() || s["in_combat"].as_bool().unwrap_or(false) {
                            quiet_since = Instant::now();
                        }
                        if quiet_since.elapsed() < QUIET {
                            continue;
                        }
                        quiet = true;
                        Input::Event { text: QUIET_EVENT.to_string(), react: true }
                    }
                },
            };
            // (anything from the game or the player breaks the quiet; a summary is the brain's own)
            if !matches!(input, Input::Summary { .. }) {
                quiet_since = Instant::now();
            }
            let result = match input {
                Input::Heard(text) => self.on_heard(&text),
                Input::Event { text, react } => {
                    // (and the events that came in meanwhile, all at once: he reacts to what's
                    // happened since, not to each in turn, falling behind)
                    let mut events = vec![(text, react)];
                    while let Ok(more) = inputs.try_recv() {
                        match more {
                            Input::Event { text, react } => events.push((text, react)),
                            other => {
                                waiting = Some(other);
                                break;
                            }
                        }
                    }
                    self.on_events(&events)
                }
                Input::Saved(id) => self.on_saved(id),
                Input::Loaded(id) => self.on_loaded(&id),
                Input::Summary { memory, folded, text } => self.on_summary(memory, folded, text),
            };
            if let Err(e) = result {
                eprintln!("[brain] error: {e:#}");
            }
            self.fold_over(if quiet { FEED_TOKENS } else { FEED_MAX_TOKENS });
        }
    }

    fn on_heard(&mut self, text: &str) -> Result<()> {
        self.link.send_json(&json!({"type": "heard", "text": text}));
        let player = self.player();
        self.memory.feed.push(format!("{player}: {text}"));
        self.trace.add("heard", json!({"text": text}));
        self.respond(true)
    }

    fn on_events(&mut self, events: &[(String, bool)]) -> Result<()> {
        for (text, react) in events {
            let text = self.fill(text);
            self.memory.feed.push(text.clone());
            // (and the game's state as it was then: what Kev and the LLM saw of it)
            self.trace.add("event", json!({"text": text, "react": react, "state": self.state()}));
        }
        if events.iter().any(|(_, react)| *react) {
            self.respond(false)?;
        }
        Ok(())
    }

    /// To what just came in (the player's line or the game's): Kev picks what the mate does,
    /// which goes in the feed, then (for the game's) whether he speaks; if so, the LLM says it,
    /// knowing what he did. The player is always answered. The game's can't have him talk while
    /// he's talking, nor until the cooldown (the game's setting) after his voice stops: Kev
    /// only picks what he does.
    fn respond(&mut self, to_player: bool) -> Result<()> {
        let cooldown = !to_player
            && self.speaking.quiet_from().map_or(true, |quiet| Instant::now() < quiet + self.link.speech_cooldown());
        if self.decide(to_player, cooldown)? {
            // (still talking: he drops what he hadn't said yet; what's playing goes on until the
            // new line's first audio takes its place, speak.rs)
            if self.speaking.is_speaking() {
                self.speaking.interrupt();
            }
            self.say()?;
        }
        Ok(())
    }

    fn saves_path(cfg: &Config) -> PathBuf {
        cfg.base_dir.join("saves.json")
    }

    fn on_saved(&mut self, id: String) -> Result<()> {
        self.saves.retain(|(saved, _)| *saved != id);
        self.saves.push((id.clone(), self.memory.clone()));
        if self.saves.len() > SAVES {
            self.saves.remove(0);
        }
        std::fs::write(Self::saves_path(&self.cfg), serde_json::to_string(&self.saves)?)?;
        self.trace.add("event", json!({"text": format!("(the game saved: {id})")}));
        // (a safe point to fold the feed)
        self.fold_over(FEED_TOKENS);
        Ok(())
    }

    /// Back to what the mate remembered at that save; nothing, for a save it never saw.
    fn on_loaded(&mut self, id: &str) -> Result<()> {
        let found = self.saves.iter().rev().find(|(saved, _)| saved == id);
        self.memory = found.map(|(_, memory)| memory.clone()).unwrap_or_default();
        self.memory_generation += 1;
        self.summarizing = false;
        if self.speaking.is_speaking() {
            self.speaking.interrupt();
            self.link.send_json(&json!({"type": "stop_audio"}));
        }
        let note = if found.is_some() { "memory restored" } else { "a save it never saw: memory cleared" };
        self.trace.add("event", json!({"text": format!("(the game loaded {id}: {note})")}));
        println!("[brain] loaded {id}: {note}");
        Ok(())
    }

    fn on_summary(&mut self, memory: u64, folded: usize, text: String) -> Result<()> {
        self.summarizing = false;
        if memory == self.memory_generation && folded <= self.memory.feed.len() {
            self.memory.feed.drain(..folded);
            self.memory.summary = text;
            self.trace.add("memory", json!({"summary": self.memory.summary, "folded": folded}));
        }
        Ok(())
    }

    /// The feed over `limit` tokens: its oldest lines, down to FEED_FOLDED_TO, are folded into the
    /// summary, on another thread (the LLM writes it), so the mate goes on meanwhile.
    fn fold_over(&mut self, limit: usize) {
        let tokens = |lines: &[String]| lines.iter().map(|line| line.len() + 1).sum::<usize>() / 4;
        if self.summarizing || tokens(&self.memory.feed) <= limit {
            return;
        }
        let mut folded = 0;
        while folded < self.memory.feed.len() && tokens(&self.memory.feed[folded..]) > FEED_FOLDED_TO {
            folded += 1;
        }
        self.summarizing = true;
        let (agent, url, name) = (self.agent.clone(), self.cfg.llm.url.clone(), self.cfg.mate.name.clone());
        let (previous, lines) = (self.memory.summary.clone(), self.memory.feed[..folded].join("\n"));
        let (inputs, memory, trace) = (self.inputs.clone(), self.memory_generation, self.trace.clone());
        let temperature = 0.3;
        std::thread::spawn(move || {
            let start = Instant::now();
            let request = format!(
                "{previous}\n{lines}\n\nRewrite all of the above as {name}'s memory of what happened: one paragraph of \
                 at most 250 words, past tense, about {name} in the third person. Keep every name, number and fact exactly \
                 as said (people, pets, places, plans, promises), and what matters for the story and the fights. Add \
                 nothing that isn't above. Leave out small talk that doesn't matter."
            );
            let messages = json!([{"role": "user", "content": request}]);
            let reply: Result<Value> = agent
                .post(&url)
                .send_json(json!({"messages": messages, "temperature": temperature}))
                .map_err(anyhow::Error::from)
                .and_then(|r| Ok(r.into_json()?));
            let output = reply.as_ref().ok().and_then(|r| r["choices"][0]["message"]["content"].as_str()).map(str::to_string);
            trace.add(
                "summary",
                json!({
                    "messages": messages,
                    "settings": {"url": url, "temperature": temperature,
                                 "model": reply.as_ref().ok().and_then(|r| r["model"].as_str().map(str::to_string))},
                    "output": output,
                    "ms": start.elapsed().as_millis() as u64,
                    "error": reply.as_ref().err().map(|e| format!("{e:#}")),
                }),
            );
            match output {
                Some(output) => {
                    let _ = inputs.send(Input::Summary { memory, folded, text: clean(&output) });
                }
                None => eprintln!("[brain] summary failed: {:#}", reply.err().map(|e| e.to_string()).unwrap_or_default()),
            }
        });
    }

    fn state(&self) -> Value {
        self.link.state.lock().unwrap().clone()
    }

    /// The player's name: player_name.txt's (beside the config), else the game's.
    fn player(&self) -> String {
        std::fs::read_to_string(self.cfg.base_dir.join("player_name.txt"))
            .ok()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .or_else(|| self.state()["player_name"].as_str().map(str::to_string))
            .unwrap_or_else(|| "the player".to_string())
    }

    /// The game's text (and the prompts) with the player's name and the game's in place of
    /// {player} and {game}.
    fn fill(&self, text: &str) -> String {
        text.replace("{player}", &self.player()).replace("{game}", self.state()["game"].as_str().unwrap_or("a game"))
    }

    /// The level's story so far, as the player knows it (levels/<level>.txt), and what the mate
    /// remembers of before the feed.
    fn context(&self) -> Vec<String> {
        let mut parts = Vec::new();
        if let Some(story) = self.state()["level"].as_str().and_then(|level| {
            std::fs::read_to_string(self.cfg.base_dir.join("levels").join(format!("{level}.txt"))).ok()
        }) {
            parts.push(self.fill(story.trim()));
        }
        if !self.memory.summary.is_empty() {
            parts.push(format!("What {} remembers: {}", self.cfg.mate.name, self.memory.summary));
        }
        parts
    }

    fn right_now(&self) -> String {
        format!("Right now: {}", self.fill(self.state()["state"].as_str().unwrap_or("unknown")))
    }

    /// Who the mate is: the system prompt.
    fn system(&self) -> String {
        self.fill(self.system.get().trim())
    }

    /// What holds right now, in both prompts just before the newest lines: the combat and radio
    /// prompts while the game says so, and the game's state. It changes all the time; before it,
    /// the prompt only grows, so the servers reuse what they read of it last time.
    fn situation(&self) -> String {
        let s = self.state();
        let mut parts = Vec::new();
        if s["in_combat"].as_bool().unwrap_or(false) {
            parts.push(self.fill(self.combat.get().trim()));
        }
        if s["on_radio"].as_bool().unwrap_or(false) {
            parts.push(self.fill(self.radio.get().trim()));
        }
        parts.push(self.right_now());
        parts.join("\n")
    }

    /// Where the feed's newest lines start: after the mate's own last line (the game's and the
    /// player's since).
    fn newest(&self) -> usize {
        let own = format!("{}: ", self.cfg.mate.name);
        self.memory.feed.iter().rposition(|line| line.starts_with(&own)).map_or(0, |at| at + 1)
    }

    /// Kev's prompt: who the mate is, the context, the feed's newest KEV_FEED_LINES, with what
    /// holds right now before the lines since the mate last spoke, so that the newest line is
    /// last: what the LLM gets, as one block.
    fn prompt(&self) -> String {
        let start = self.memory.feed.len().saturating_sub(KEV_FEED_LINES);
        let newest = self.newest().max(start);
        let mut parts = vec![self.system()];
        parts.extend(self.context());
        if newest > start {
            parts.push(self.memory.feed[start..newest].join("\n"));
        }
        parts.push(self.situation());
        if newest < self.memory.feed.len() {
            parts.push(self.memory.feed[newest..].join("\n"));
        }
        parts.join("\n\n")
    }

    /// The LLM's: the same, as chat turns, which the server puts in the loaded model's own
    /// template. The system turn is who the mate is; then the context, the game's lines and the
    /// player's as the user's turns and the mate's own lines as its turns (a line or more in a
    /// row, one turn), what holds right now just before the newest lines. As one block of text,
    /// the model wrote a scene to carry on (a line, then another, each with its own voice) rather
    /// than the mate's turn.
    fn messages(&self) -> Vec<Value> {
        let mut turns: Vec<(&str, String)> = vec![("system", self.system())];
        let mut add = |role: &'static str, text: &str| match turns.last_mut() {
            Some((last, joined)) if *last == role => {
                joined.push('\n');
                joined.push_str(text);
            }
            _ => turns.push((role, text.to_string())),
        };
        let context = self.context().join("\n\n");
        if !context.is_empty() {
            add("user", &context);
        }
        let own = format!("{}: ", self.cfg.mate.name);
        let newest = self.newest();
        for line in &self.memory.feed[..newest] {
            match line.strip_prefix(&own) {
                Some(said) => add("assistant", said),
                None => add("user", line),
            }
        }
        add("user", &self.situation());
        for line in &self.memory.feed[newest..] {
            add("user", line);
        }
        let messages: Vec<Value> = turns.iter().map(|(role, text)| json!({"role": role, "content": text})).collect();
        self.trace.set_prompt(
            &turns.iter().map(|(role, text)| format!("[{role}]\n{text}")).collect::<Vec<_>>().join("\n\n"),
        );
        messages
    }

    /// Kev picks what the mate does, then whether he speaks (returned; not asked of the player's
    /// line, which is answered, nor in the cooldown). What he does goes in the feed ("Kevin
    /// stands up."), for the LLM to see before it says anything. An action that reports back (a
    /// search) puts what it found there too, and Kev picks again from what it offers to do
    /// about it (its "then").
    fn decide(&mut self, to_player: bool, cooldown: bool) -> Result<bool> {
        let name = self.cfg.mate.name.clone();
        let start = Instant::now();
        // (each call to Kev, with what went in and came out, for the history page)
        let mut calls = Vec::new();
        let actions = self.state()["actions"].as_array().cloned().unwrap_or_default();
        let done = self.act("do", &actions, &mut calls)?;
        let then = actions
            .iter()
            .find(|a| a["name"] == done["action"])
            .and_then(|a| a["then"].as_array())
            .filter(|then| !then.is_empty())
            .cloned();
        let then_done = match then {
            Some(then) => self.act("then", &then, &mut calls)?,
            None => Value::Null,
        };
        // (then whether he speaks, with what he just did in the feed; Kev's yes or no: yes when
        // it's the likelier)
        let ask_speak = !to_player && !cooldown;
        let speak = to_player || ask_speak && {
            let speak_question = json!({"speak": {"type": "noul", "instructions": format!("Should {name} speak now?")}});
            let spoken = self.kev("speak", &self.prompt(), &speak_question, &mut calls)?;
            spoken["speak"]["noul"].as_f64().unwrap_or(0.0) >= 0.5
        };
        self.trace.add(
            "kev",
            json!({
                "calls": calls,
                "action": done["action"],
                "option": done["option"],
                "sent": done["action"] != "nothing",
                "seen": done["seen"],
                // (what he did about what an action found, if it offered anything)
                "then": then_done,
                "speak": speak,
                "speak_asked": ask_speak,
                // (not asked: he's talking, or his voice stopped under the cooldown ago)
                "cooldown": cooldown,
                "ms": start.elapsed().as_millis() as u64,
            }),
        );
        let did = |done: &Value| {
            format!(
                "{}{}",
                done["action"].as_str().unwrap_or("nothing"),
                done["option"].as_str().map(|o| format!(" {o}")).unwrap_or_default()
            )
        };
        println!(
            "[kev] {}{}, speak {} ({} ms)",
            did(&done),
            if then_done.is_null() { String::new() } else { format!(", then {}", did(&then_done)) },
            if cooldown { "not asked (cooldown)" } else if speak { "yes" } else { "no" },
            start.elapsed().as_millis()
        );
        Ok(speak)
    }

    /// Kev picks one of `actions` (or nothing), and for one with targets, which. It goes to the
    /// game and in the feed, with what it found (its "result"), or, for one of the player's view
    /// (its "view"), what the mate makes of that. What was done: its `action`, `option`, `seen`.
    fn act(&mut self, purpose: &str, actions: &[Value], calls: &mut Vec<Value>) -> Result<Value> {
        let name = self.cfg.mate.name.clone();
        let mut options = serde_json::Map::new();
        options.insert("nothing".to_string(), json!(format!("{name} carries on as he is")));
        for a in actions {
            let description = self.fill(a["description"].as_str().unwrap_or_default());
            options.insert(a["name"].as_str().unwrap_or_default().to_string(), json!(format!("{name} {description}")));
        }
        let prompt = self.prompt();
        let question = json!({"do": {"type": "choice", "instructions": format!("Which of these does {name} do now?"), "criteria": options}});
        let answers = self.kev(purpose, &prompt, &question, calls)?;
        let action = answers["do"]["choice"].as_str().unwrap_or("nothing").to_string();
        if action == "nothing" {
            return Ok(json!({"action": action}));
        }
        let chosen = actions.iter().find(|a| a["name"] == action.as_str()).cloned().unwrap_or_default();
        let mut msg = json!({"type": "action", "action": action});
        // ("Kevin stands up." / "Kevin shoots at an enemy: grunt close (8 m) ahead.")
        let mut line = options[action.as_str()].as_str().unwrap_or_default().to_string();
        if let Some(targets) = chosen["options"].as_object().filter(|o| !o.is_empty()) {
            let which = json!({"which": {"type": "choice", "instructions": format!("{line}. Which one?"), "criteria": targets}});
            let which_purpose = if purpose == "do" { "which".to_string() } else { format!("{purpose}: which") };
            let answer = self.kev(&which_purpose, &prompt, &which, calls)?;
            msg["option"] = answer["which"]["choice"].clone();
            if let Some(label) = msg["option"].as_str().and_then(|id| targets.get(id)).and_then(Value::as_str) {
                line = format!("{line}: {label}");
            }
        }
        let mut seen = Value::Null;
        match chosen["view"].as_str() {
            Some(view) => {
                let (what, picture) = self.look(view, &mut msg)?;
                line = format!("{line}: {what}");
                seen = json!({"what": what, "picture": picture});
            }
            None => {
                self.link.send_json(&msg);
                if let Some(result) = chosen["result"].as_str() {
                    line = format!("{line}: {}", self.fill(result));
                }
            }
        }
        self.memory.feed.push(format!("{line}."));
        Ok(json!({"action": action, "option": msg["option"], "seen": seen}))
    }

    /// The mate's eyes: the player's view from the game, and what the LLM makes of it, as his
    /// eyes, not as him (what's there, plainly, not a line of his), from the game's state and the
    /// feed's newest lines. "describe": what's there; "choose": which of the spots marked on it
    /// he goes to (sent to the game as the action's option). What goes in the feed, and the
    /// picture's name (logs/views/<name>.png, kept for the history page).
    fn look(&mut self, kind: &str, msg: &mut Value) -> Result<(String, Value)> {
        let (name, player) = (self.cfg.mate.name.clone(), self.player());
        let start = Instant::now();
        let Some(mut view) = self.link.view(Duration::from_secs(2)) else {
            self.trace.add("sight", json!({"kind": kind, "error": "the game sent no view"}));
            return Ok(("he can't see it".to_string(), Value::Null));
        };
        let choose = kind == "choose";
        if choose {
            picture::mark(&mut view.rgb, view.width, view.height, &view.spots);
        }
        let png = picture::png(&view.rgb, view.width, view.height);
        let file = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis().to_string();
        let views = self.cfg.base_dir.join("logs").join("views");
        let _ = std::fs::create_dir_all(&views);
        let _ = std::fs::write(views.join(format!("{file}.png")), &png);

        let eyes = format!("You're the eyes of {name}, {player}'s teammate in Halo: Combat Evolved: you're shown what {player} is looking at.");
        let (system, question) = if choose {
            (
                format!("{eyes} Numbered spots are marked on the ground in it, and you pick the one {name} goes to, from what's been said."),
                format!("Here's what {player} is looking at. Which spot does {name} go to? Answer with \"place\": what's at that spot, in a few words, and \"spot\": its number (0 if none fits)."),
            )
        } else {
            (
                format!("{eyes} Say plainly what's there, in a sentence or two: what you see, not anyone's lines."),
                format!("Here's what {player} is looking at. What's there?"),
            )
        };
        let newest = self.memory.feed[self.memory.feed.len().saturating_sub(12)..].join("\n");
        let question = format!("{}\n\nWhat's been happening, newest last:\n{newest}\n\n{question}", self.right_now());
        let mut request = json!({
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": ""}],
            "temperature": self.cfg.llm.temperature,
            "stream": false,
        });
        if choose {
            let spots: Vec<u64> = std::iter::once(0).chain(view.spots.iter().filter_map(|s| s["id"].as_u64())).collect();
            request["response_format"] = json!({"type": "json_schema", "json_schema": {"name": "spot", "schema": {
                "type": "object",
                "properties": {"place": {"type": "string", "maxLength": 80}, "spot": {"type": "integer", "enum": spots}},
                "required": ["place", "spot"],
            }}});
        }
        // (the picture in the request, and as traced: by its file, not its data)
        let mut traced = request.clone();
        for (request, url) in [(&mut request, format!("data:image/png;base64,{}", picture::base64(&png))), (&mut traced, format!("/view?name={file}"))] {
            request["messages"][1]["content"] =
                json!([{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": url}}]);
        }
        let output = match self.agent.post(&self.cfg.llm.url).send_json(&request) {
            Ok(response) => response.into_json::<Value>()?["choices"][0]["message"]["content"].as_str().unwrap_or_default().trim().to_string(),
            Err(e) => {
                let error = match e {
                    ureq::Error::Status(code, response) => format!("{code}: {}", response.into_string().unwrap_or_default()),
                    other => other.to_string(),
                };
                eprintln!("[sight] {kind} failed: {error}");
                self.trace.add("sight", json!({"kind": kind, "request": traced, "picture": file, "error": error}));
                return Ok(("he can't see it".to_string(), json!(file)));
            }
        };
        let what = if choose {
            let answer: Value = serde_json::from_str(&output).unwrap_or_default();
            let spot = answer["spot"].as_u64().unwrap_or(0);
            match view.spots.iter().find(|s| spot > 0 && s["id"].as_u64() == Some(spot)) {
                Some(found) => {
                    msg["option"] = json!(spot.to_string());
                    self.link.send_json(msg);
                    format!("{} ({} m off)", answer["place"].as_str().unwrap_or("there").trim(), found["metres"])
                }
                None => "none that fits".to_string(),
            }
        } else {
            clean(&output)
        };
        self.trace.add(
            "sight",
            json!({
                "kind": kind,
                "url": self.cfg.llm.url,
                "request": traced,
                "output": output,
                "seen": what,
                "picture": file,
                "spots": view.spots.len(),
                "ms": start.elapsed().as_millis() as u64,
            }),
        );
        println!("[sight] {kind}: {what} ({} ms)", start.elapsed().as_millis());
        Ok((what, json!(file)))
    }

    /// Asks Kev the questions about `input`; the call, what went in and came out, goes in `calls`.
    fn kev(&self, purpose: &str, input: &str, questions: &Value, calls: &mut Vec<Value>) -> Result<Value> {
        let start = Instant::now();
        let request = json!({"model": "kev-latest", "state": input, "questions": questions});
        let resp: Value = self.agent.post(&self.cfg.kev.url).send_json(&request)?.into_json()?;
        calls.push(json!({
            "purpose": purpose,
            "url": self.cfg.kev.url,
            "request": request,
            "response": resp,
            "ms": start.elapsed().as_millis() as u64,
        }));
        Ok(resp["answers"].clone())
    }

    /// Streams an LLM line for the mate, speaking each sentence as soon as it's complete, and
    /// adds it to the feed.
    fn say(&mut self) -> Result<()> {
        let name = self.cfg.mate.name.clone();
        let messages = self.messages();
        let start = Instant::now();
        let generation = self.speaking.generation();
        let request = json!({"messages": messages, "temperature": self.cfg.llm.temperature, "stream": true});
        let resp = self.agent.post(&self.cfg.llm.url).send_json(&request)?;
        // (he's talking from here until speak.rs has voiced the End below: always sent)
        self.speaking.begin_line();
        let mut failed = None;
        // (for the history page: the model the server named, and when its first words came)
        let mut model = Value::Null;
        let mut first_token_ms = Value::Null;

        let mut full = String::new();
        let mut notes = Notes::default();
        let mut splitter = SentenceSplitter::default();
        // How he says it: the latest "[...]" note, for the voice; each holds until the next.
        let mut how = String::new();
        let mut directions: Vec<String> = Vec::new();
        let speak = |sentence: String, how: &str| {
            let sentence = sentence.trim().to_string();
            if !sentence.is_empty() {
                let _ = self.speech.send(SpeechJob::Sentence(sentence, how.to_string(), generation));
            }
        };
        for line in BufReader::new(resp.into_reader()).lines() {
            if self.speaking.generation() != generation {
                // The player started talking; drop the rest.
                break;
            }
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    failed = Some(anyhow::Error::from(e));
                    break;
                }
            };
            let Some(data) = line.strip_prefix("data: ") else { continue };
            if data.trim() == "[DONE]" {
                break;
            }
            let chunk: Value = match serde_json::from_str(data) {
                Ok(chunk) => chunk,
                Err(e) => {
                    failed = Some(anyhow::Error::from(e));
                    break;
                }
            };
            if model.is_null() {
                model = chunk["model"].clone();
            }
            if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
                if first_token_ms.is_null() {
                    first_token_ms = json!(start.elapsed().as_millis() as u64);
                }
                full.push_str(delta);
                for piece in notes.push(delta) {
                    match piece {
                        Piece::Text(text) => {
                            for sentence in splitter.push(&text) {
                                speak(sentence, &how);
                            }
                        }
                        Piece::Note(note) => {
                            // (what came before it is said as it was: a note mid-sentence splits it)
                            if let Some(rest) = splitter.finish() {
                                speak(rest, &how);
                            }
                            how = note;
                            directions.push(how.clone());
                        }
                    }
                }
            }
        }
        if let Some(rest) = splitter.finish() {
            speak(rest, &how);
        }
        let _ = self.speech.send(SpeechJob::End(generation));
        // (his turn as he wrote it, every note and paragraph: as the LLM sees its own lines)
        let line = full.trim().to_string();
        println!("[llm] {name}: \"{line}\" ({} ms)", start.elapsed().as_millis());
        self.trace.add(
            "llm",
            json!({
                "url": self.cfg.llm.url,
                "model": model,
                "request": request,
                "output": full,
                "directions": directions,
                "line": line,
                "first_token_ms": first_token_ms,
                "ms": start.elapsed().as_millis() as u64,
                "cut_off": self.speaking.generation() != generation,
                "error": failed.as_ref().map(|e| format!("{e:#}")),
            }),
        );
        self.memory.feed.push(format!("{name}: {line}"));
        failed.map_or(Ok(()), Err)
    }
}

/// Splits streamed text into speakable sentences.
#[derive(Default)]
struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            let chars: Vec<(usize, char)> = self.buf.char_indices().collect();
            let mut cut = None;
            for w in chars.windows(2) {
                let ((i, c), (_, next)) = (w[0], w[1]);
                let ends = matches!(c, '.' | '!' | '?' | '…') && next.is_whitespace() || c == '\n';
                if ends {
                    cut = Some(i + c.len_utf8());
                    break;
                }
            }
            let Some(cut) = cut else { break };
            let sentence = clean(&self.buf[..cut]);
            self.buf.drain(..cut);
            if !sentence.is_empty() {
                out.push(sentence);
            }
        }
        out
    }

    fn finish(&mut self) -> Option<String> {
        let rest = clean(&std::mem::take(&mut self.buf));
        (!rest.is_empty()).then_some(rest)
    }
}

/// The sounds the voice makes (Breeze TTS's four), by the ways the LLM writes them in a line:
/// "(sighs)" is said as "(sigh)". Anything else in round brackets isn't said.
const SOUNDS: [(&str, &str); 19] = [
    ("laugh", "(laugh)"), ("laughs", "(laugh)"), ("laughing", "(laugh)"), ("laughter", "(laugh)"),
    ("chuckle", "(laugh)"), ("chuckles", "(laugh)"), ("chuckling", "(laugh)"), ("giggles", "(laugh)"),
    ("snickers", "(laugh)"), ("cackles", "(laugh)"),
    ("sigh", "(sigh)"), ("sighs", "(sigh)"), ("sighing", "(sigh)"),
    ("cough", "(cough)"), ("coughs", "(cough)"), ("coughing", "(cough)"),
    ("clears throat", "(clears throat)"), ("clears his throat", "(clears throat)"), ("clearing throat", "(clears throat)"),
];

/// A piece of the LLM's streamed line: what he says, or a "[how he says it]" note, which holds
/// for what follows until the next.
enum Piece {
    Text(String),
    Note(String),
}

/// Splits the streamed line into its pieces, holding back a note until it's closed.
#[derive(Default)]
struct Notes {
    buf: String,
}

impl Notes {
    fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut pieces = Vec::new();
        loop {
            let Some(open) = self.buf.find('[') else {
                if !self.buf.is_empty() {
                    pieces.push(Piece::Text(std::mem::take(&mut self.buf)));
                }
                break;
            };
            if open > 0 {
                pieces.push(Piece::Text(self.buf[..open].to_string()));
                self.buf.drain(..open);
            }
            let Some(close) = self.buf.find(']') else { break };
            pieces.push(Piece::Note(self.buf[1..close].trim().to_string()));
            self.buf.drain(..=close);
        }
        pieces
    }
}

/// Strips things TTS shouldn't read out: markdown, emoji, wrapping quotes.
fn clean(text: &str) -> String {
    // Drop stage directions like "[looks around]" or "(nods)", but not the sounds the voice
    // makes; an unclosed one runs to the end.
    let mut depth = 0u32;
    let mut unbracketed = String::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if depth == 0 && c == '(' {
            let sound = rest[1..].find(')').and_then(|end| {
                let written = rest[1..1 + end].trim().to_lowercase();
                SOUNDS.iter().find(|(as_written, _)| *as_written == written).map(|(_, said)| (said, end + 2))
            });
            if let Some((said, length)) = sound {
                unbracketed.push_str(said);
                rest = &rest[length..];
                continue;
            }
        }
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => unbracketed.push(c),
            _ => {}
        }
        rest = &rest[c.len_utf8()..];
    }
    let filtered: String = unbracketed
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '#' | '~' | '`'))
        .filter(|&c| !(('\u{1F000}'..='\u{1FAFF}').contains(&c) || ('\u{2600}'..='\u{27BF}').contains(&c)))
        .collect();
    let collapsed = filtered.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim_matches(|c: char| c == '"' || c == '\u{201C}' || c == '\u{201D}').trim();
    if trimmed.chars().any(char::is_alphanumeric) {
        trimmed.to_string()
    } else {
        String::new()
    }
}
