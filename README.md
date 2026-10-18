# m8

A voice brain for AI companions in games. You talk to your mate out loud; it hears the game,
decides what to do, does it, and answers in a voice of its own.

m8 knows nothing about any one game. The game tells it what's going on, in words, and what the
mate can do right now; m8 sends back what the mate does, and its voice. Hook up any game that
can open a socket, plug in the models you like, write a persona.

> **Not a release.** This is a working setup shared as it is, for power users to pull, fork and
> bend to their own game and models. It runs on Windows with an NVIDIA GPU, starts with
> PowerShell scripts, and expects you to read the code.

**Supported game: Halo: Combat Evolved**, as a mod of [OpenCE](https://github.com/OpenCommunityEdition/OpenCE),
the port of the Xbox game ([games/halo](games/halo)). More games can be added the same way. The persona that comes
with it is **Kevin**, a Spartan ([examples/kevin](examples/kevin)).

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

## The models are endpoints

m8 runs no models of its own (bar Silero VAD, a 2 MB model that finds where speech starts and
ends in the mic's audio). Each job is a server it calls, set in `m8.toml`; run them where you
like, local or not. It speaks the APIs of what I used:

| Job | API m8 calls | What I ran behind it |
| --- | --- | --- |
| Decide | llama.cpp's `/v1/systemone` (`[kev] url`) | [Kev-4B](https://huggingface.co/ggml-org/Kev-4B-GGUF) Q8, a decision model: it scores options, it doesn't write. Its server reads its whole prompt every call, so m8 gives it only the feed's newest 20 lines |
| Talk and see | OpenAI-style `/v1/chat/completions` (`[llm] url`), images in the messages to see | [Gemma 4 12B QAT, uncensored by HauhauCS](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced) Q4_K_M with its mmproj, on llama.cpp |
| Speak | Breeze TTS 2's `/v1/audio/speech`, streamed (`voice_server`) | [Breeze TTS 2](https://huggingface.co/BreezeBlue/breeze-tts-2), cloning a voice clip, on [breeze-tts](https://github.com/breezeblue-ai/breeze-tts) in WSL |
| Hear | OpenAI-style `/v1/audio/transcriptions` (`[stt] url`) | [Parakeet TDT 0.6B v2](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) int8, on `stt-server` (built with m8) |

Using something else (another API, half local and half hosted): each call is in one place to
change. Deciding is `Brain::kev` (`brain.rs`), talking `Brain::say`, seeing `Brain::look`,
speaking `Voice::stream` (`speak.rs`) and hearing `Transcriber::transcribe` (`listen.rs`).

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

## Playing it in Halo

The mod is m8's changes to OpenCE ([games/halo](games/halo)); you bring the models (Setup,
below) and your own copy of the game.

1. Get the game and put the mod on it: `games/halo/setup.ps1` clones OpenCE into `halo/` and
   applies the changes.
2. Build it, on Windows, in `halo/` (Python, [ninja](https://ninja-build.org/) and LLVM with
   clang 22 or later):

   ```bash
   python configure.py --release
   ninja windows
   ```

   That makes `halo/build/windows/halo.exe`.
3. The game data: start `halo.exe` once and give it an Xbox disc image of Halo: Combat Evolved,
   as OpenCE's README ("Game data") says; it extracts the maps.
4. Set up m8 (Setup, below), then run `start-all.ps1`: it starts the models, m8 and the game,
   which connects to m8 by itself.
5. In a campaign level the teammate appears beside you. Talk to him; **M** mutes the mic
   (an icon shows it); typing `kevin <text>` in the game's console gives him a line as a game
   master; **KEVIN SETUP** on the profile menu sets his speech cooldown, his volume beside you
   and over the radio, and how far off he goes to the radio.

What the mod changes: [games/halo/README.md](games/halo/README.md).

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
   release: `silero_vad.onnx` into `models/` (m8's VAD), and, for the speech-to-text server,
   `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2` unpacked into `models/`.

2. Breeze, in WSL: clone breeze-tts into `~/breeze-tts`, set it up as its README says (its
   `.venv`), then apply this repo's patch, which encodes the voice clip once instead of with every
   line:

   ```bash
   git -C ~/breeze-tts am /mnt/<path to this repo>/patches/breeze-tts-reference-cache.patch
   ```

3. A voice clip and its transcript in `voices/` ([voices/README.md](voices/README.md)).
4. Your name, as the mate calls you: one line in `player_name.txt`.
5. Build m8 (and `stt-server`, beside it):

   ```bash
   cd m8
   cargo build --release
   ```

Then `start-all.ps1` starts it all, each in its own window: the decision model
(`start-kev.ps1`), the LLM (`start-chat.ps1`), Breeze (`start-breeze.ps1`; its first start takes
about 4 minutes), speech to text (`start-stt.ps1`), m8 once they're up, and the game if it's
built in `halo/`. Change the scripts for your own servers and game.

## Watching it

- **Brain page**, http://127.0.0.1:8090: the prompt as the models get it, every decision and
  line as it happens, buttons to send test events, and the persona's prompts to edit.
- **History page**, http://127.0.0.1:8090/history: every model call of a session, exactly what
  went in and what came out, with the pictures the mate saw. Each session is kept in `logs/`.
- Type a line in m8's window to say it without a mic.

## The code

`m8/src`: `brain.rs` (the feed, decisions, the LLM, the mate's eyes, memory), `listen.rs` (the
mic, Silero, speech to text), `speak.rs` (the voice), `link.rs` (the game), `site.rs` (the
pages), `picture.rs` (the view's numbered spots), `config.rs` (`m8.toml`), and
`bin/stt-server.rs` (Parakeet behind the transcriptions endpoint).

## Licence

MIT. The models, Breeze and llama.cpp have their own licences.
