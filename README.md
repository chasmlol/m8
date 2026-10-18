# m8

A voice brain for AI companions in games. You talk to your mate out loud; it hears the game,
decides what to do, does it, and answers in a voice of its own.

m8 knows nothing about any one game. The game tells it what's going on, in words, and what the
mate can do right now; m8 sends back what the mate does, and its voice. Hook up any game that
can open a socket, plug in the models you like, write a persona.

> **Not a release.** This is a working setup shared as it is, for power users to pull, fork and
> bend to their own game and models. It runs on Windows with an NVIDIA GPU, starts with
> PowerShell scripts, and expects you to read the code.

The example it was built with is **Kevin**, a Spartan teammate in a port of Halo: Combat Evolved
([examples/kevin](examples/kevin)). The game side of that isn't in this repo.

## How a turn goes

1. Something comes in: you said something (mic, or typed), or the game sent an event.
2. It goes in the feed: everything said and done, the mate's own actions too.
3. The decision model picks one of the actions the game offers (or none), and its target. An
   action can report back (a search: what it found goes in the feed, and the model picks again
   from what that offers), or need the player's view (the LLM looks at a frame from the game:
   says what's there, or picks one of the spots marked on the ground).
4. For the game's events, the decision model says whether the mate speaks. Your lines are always
   answered. After the mate stops talking, the game's events can't start it again until the
   cooldown (the game sends it).
5. The LLM writes the line, with how to say it ("[yelling over gunfire]") and sounds
   ("(laugh)"); the TTS says it, sentence by sentence; the game plays it.

Memory: the feed keeps about 16k tokens; past that, its oldest lines are folded into a summary
at a quiet moment or a save. The game's saves carry the memory: loading a save brings back what
the mate knew then.

## The models: what I used, what you can plug in

| Job | What I used | What else works |
| --- | --- | --- |
| Decide | [Kev-4B](https://huggingface.co/ggml-org/Kev-4B-GGUF) Q8, a decision model (it scores options, it doesn't write), on llama.cpp's `/v1/systemone` | Any decision model llama.cpp serves on `/v1/systemone` (`[kev] url`). It reads its whole prompt every call, so m8 gives it only the feed's newest 20 lines |
| Talk and see | [Gemma 4 12B QAT, uncensored by HauhauCS](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced) Q4_K_M with its mmproj, on llama.cpp | Any server and model behind an OpenAI-style `/v1/chat/completions` (`[llm] url`); the server applies the model's own chat template. Seeing needs a model that takes images; without, looking just fails |
| Speak | [Breeze TTS 2](https://huggingface.co/BreezeBlue/breeze-tts-2), cloning a voice clip, on [breeze-tts](https://github.com/breezeblue-ai/breeze-tts) in WSL | Any server with Breeze's `/v1/audio/speech` API (`voice_server`), or Pocket TTS's built-in voices with no server at all |
| Hear | [Parakeet TDT 0.6B v2](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) int8 and Silero VAD, on sherpa-onnx inside m8 | Another NeMo transducer in sherpa-onnx's format (`parakeet_dir`) |

## Plugging in a game

The game connects to m8 at `127.0.0.1:47800` (`[game] port`) and reconnects whenever m8
restarts. Frames are `[u8 kind][u32 big-endian length][payload]`:

| Kind | Way | Payload |
| --- | --- | --- |
| 1 | both | a JSON message |
| 2 | to the game | the mate's voice: mono 16-bit samples, at the `hello` message's rate |
| 3 | to m8 | dialogue the game played, to hear: `[u16 length][JSON: speaker, radio, react, sample_rate][mono 16-bit samples]` |
| 4 | to m8 | the player's view, when asked: `[u16 length][JSON: width, height, spots][RGB rows]` |

The game sends a `state` whenever things change (Kevin's sent one every tick):

```json
{"type": "state", "game": "Halo: Combat Evolved", "level": "a30", "in_combat": false, "on_radio": false,
 "state": "Kevin: unhurt, shields full, holding a pistol (12 rounds loaded), following {player}. {player}: close by ahead. Enemies: a grunt close (8 m) ahead.",
 "actions": [
   {"name": "follow", "description": "follows {player}"},
   {"name": "shoot", "description": "shoots at an enemy", "options": {"12": "grunt close (8 m) ahead"}},
   {"name": "search_weapons", "description": "looks around for weapons", "result": "shotgun (4 m) ahead",
    "then": [{"name": "pick_up", "description": "picks up a weapon", "options": {"88": "shotgun (4 m) ahead"}}]},
   {"name": "look", "description": "takes a look at something {player} wants him to see", "view": "describe"},
   {"name": "go_to", "description": "goes to a spot {player} can see", "view": "choose"}
 ]}
```

- `state` is read as words, so say it the way you'd tell someone. `{player}` becomes the
  player's name (`player_name.txt`).
- Each action is a line the decision model picks from ("Kevin follows Chasm"). `options` are its
  targets; `result` is what doing it finds, and `then` what can be done about that; `view` asks
  for the player's view (`describe`: say what's there; `choose`: pick one of its `spots`).
- m8 answers with `{"type": "action", "action": "shoot", "option": "12"}`.

Also from the game: `event` (`text`, and `react`: false to only note it), `saved` and `loaded`
(with the save's `id`), `direct` (a game master's line), `mic` (`muted`), `settings`
(`speech_cooldown`). From m8: `hello`, `say` (a line as it starts), `heard` (what you said),
`view` (send the view), `stop_audio`, `audio_end`.

## A persona

A persona is a folder (`persona` in `m8.toml`; `name` is what the mate is called):
`system_prompt.txt`, and if you like `combat_prompt.txt`, `radio_prompt.txt` (added while the
state says `in_combat` or `on_radio`) and `levels/<level>.txt` (the story so far, by the state's
`level`). See [examples/kevin](examples/kevin).

## Setup (as I ran it)

You need an NVIDIA GPU (the three servers took about 25 GB of video memory, on an RTX 5090 with
32 GB), a CUDA build of [llama.cpp](https://github.com/ggml-org/llama.cpp) with `/v1/systemone`
([pull request 29818](https://github.com/ggml-org/llama.cpp/pull/29818); this ran on build
b11580) in `llama.cpp/`, WSL with Ubuntu for Breeze, and Rust (rustup; `rust-toolchain.toml`
picks the version).

1. The models, into `models/` (the scripts and `m8.toml` expect these names):

   ```bash
   hf download ggml-org/Kev-4B-GGUF Kev-4B-Q8_0.gguf --local-dir models/kev-4b
   hf download HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf --local-dir models/gemma-4-12b-hauhaucs
   hf download BreezeBlue/breeze-tts-2 --local-dir models/breeze-tts-2
   ```

   And from sherpa-onnx's [asr-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
   release: `silero_vad.onnx` into `models/`, and
   `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2` unpacked into `models/`.

2. Breeze, in WSL: clone breeze-tts into `~/breeze-tts`, set it up as its README says (its
   `.venv`), then apply this repo's patch, which encodes the voice clip once instead of with every
   line:

   ```bash
   git -C ~/breeze-tts am /mnt/<path to this repo>/patches/breeze-tts-reference-cache.patch
   ```

3. A voice clip and its transcript in `voices/` ([voices/README.md](voices/README.md)).
4. Your name, as the mate calls you: one line in `player_name.txt`.
5. Build m8:

   ```bash
   cd m8
   cargo build --release
   ```

Then `start-all.ps1` starts it all, each in its own window: the decision model, the LLM, Breeze
(its first start takes about 4 minutes), m8 once they're up, and a game if one is built in
`halo/`. Change the scripts for your own models and game.

## Watching it

- **Brain page**, http://127.0.0.1:8090: the prompt as the models get it, every decision and
  line as it happens, buttons to send test events, and the persona's prompts to edit.
- **History page**, http://127.0.0.1:8090/history: every model call of a session, exactly what
  went in and what came out, with the pictures the mate saw. Each session is kept in `logs/`.
- Type a line in m8's window to say it without a mic.

## The code

`m8/src`: `brain.rs` (the feed, decisions, the LLM, the mate's eyes, memory), `listen.rs` (the
mic), `speak.rs` (the voice), `link.rs` (the game), `site.rs` (the pages), `picture.rs` (the
view's numbered spots), `config.rs` (`m8.toml`).

## Licence

MIT. The models, Breeze and llama.cpp have their own licences.
