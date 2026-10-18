# AgenticNPC: mate

Kevin, an AI teammate for games. You talk to him out loud; he hears the game, decides what to
do, does it, and answers in a cloned voice. `mate` is his brain: a game connects to it on this
machine and sends what's going on; mate sends back what Kevin does and his voice.

The game side (the Halo: Combat Evolved port Kevin was built in) isn't in this repo.

## The models

| Part | Model | What it does | Served by |
| --- | --- | --- | --- |
| Decisions | [Kev-4B](https://huggingface.co/ggml-org/Kev-4B-GGUF) (Q8) | Every choice: what Kevin does and at whom, what he does about what a search found, and whether he speaks. A decision model: it scores options instead of writing text | llama.cpp, port 8080 (`/v1/systemone`) |
| Words and eyes | [Gemma 4 12B QAT, uncensored by HauhauCS](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced) (Q4_K_M + its mmproj) | Writes his lines, with how to say them ("[yelling over gunfire]") and sounds ("(laugh)"). Sees the player's view: says what's there, picks a spot to go to. Folds old memory into a summary | llama.cpp, port 8081 |
| Voice | [Breeze TTS 2](https://huggingface.co/BreezeBlue/breeze-tts-2) | Clones the voice in `voices/`, follows each direction, a sentence at a time | [breeze-tts](https://github.com/breezeblue-ai/breeze-tts) in WSL, port 7860 |
| Hearing | [Parakeet TDT 0.6B v2](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) (int8) and [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) | Hears the mic, and the game's dialogue (the radio, cutscenes) | sherpa-onnx, inside mate |

## How a turn goes

1. Something comes in: you said something (mic or typed), or the game sent an event.
2. It goes in the feed: everything that's been said and has happened, his actions too.
3. Kev picks an action from the ones the game offers (or none), and its target. An action that
   reports back (a search) adds what it found, and Kev picks again from what that offers. One of
   the player's view (look, go to a spot) gets Gemma's eyes on a frame from the game.
4. For the game's events, Kev says whether he speaks. Your lines are always answered. After he
   stops talking, the game's events can't start him again until the cooldown (the game's setting).
5. Gemma writes the line from the whole feed; Breeze says it, sentence by sentence; the game plays
   it, in the world beside you or over the radio.

Memory: the feed keeps about 16k tokens; past that, its oldest lines are folded into a summary
at a quiet moment or a save. The game's saves carry his memory: loading a save brings back what
he knew then. Kev reads only the feed's newest 20 lines (its server reads the whole prompt each
call), Gemma all of it.

## What you need

- Windows with an NVIDIA GPU. The three servers take about 25 GB of video memory (it was built on
  an RTX 5090, 32 GB).
- A CUDA build of [llama.cpp](https://github.com/ggml-org/llama.cpp) with the `/v1/systemone`
  endpoint Kev needs ([pull request 29818](https://github.com/ggml-org/llama.cpp/pull/29818);
  this was run with build b11580), in `llama.cpp/`.
- WSL with Ubuntu, for Breeze.
- Rust (rustup: `rust-toolchain.toml` picks the version).
- A voice clip you have the right to use: see `voices/README.md`.

## Setup

1. The models, into `models/` (the scripts and `mate.toml` expect these names):

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

3. Your voice clip and its transcript: `voices/soldier.wav` and `voices/soldier.txt`.
4. Your name, as Kevin calls you: one line in `player_name.txt`.
5. Build mate:

   ```bash
   cd mate
   cargo build --release
   ```

## Running

`start-all.ps1` starts everything, each in its own window: Kev, Gemma, Breeze (its first start
takes about 4 minutes), then mate once they're up.

- **Brain page**, http://127.0.0.1:8090: the prompt as the models get it, every decision and
  line as it happens, and the prompts to edit.
- **History page**, http://127.0.0.1:8090/history: every model call of a session, exactly what
  went in and what came out, with the pictures his eyes saw. Each session is kept in `logs/`.
- Type a line in mate's window to say it without a mic.

## The files

| File | What it is |
| --- | --- |
| `system_prompt.txt` | Who Kevin is |
| `combat_prompt.txt`, `radio_prompt.txt` | Added while the game says he's in a fight, or on the radio |
| `levels/<level>.txt` | Each level's story so far, for context |
| `mate.toml` | The servers, the voice, the mic |
| `player_name.txt` | Your name (not in the repo) |
| `mate/` | The program: `brain.rs` (the feed, Kev, the LLM, memory), `listen.rs` (the mic), `speak.rs` (the voice), `link.rs` (the game), `site.rs` (the pages), `picture.rs` (the view's marked spots) |

## The game link

A game connects to `127.0.0.1:47800`. Frames are `[u8 kind][u32 big-endian length][payload]`:

| Kind | Way | Payload |
| --- | --- | --- |
| 1 | both | a JSON message |
| 2 | to the game | Kevin's voice: mono 16-bit samples at the `hello` message's rate |
| 3 | to mate | dialogue the game played, to hear: `[u16 length][JSON: speaker, radio, react, sample_rate][mono 16-bit samples]` |
| 4 | to mate | the player's view, when asked: `[u16 length][JSON: width, height, spots][RGB rows]` |

From the game: `state` (the situation in words and the actions Kevin can take now, each with a
description and maybe targets, a `result` to report or a `then` to follow up with, or a `view`),
`event`, `saved` and `loaded` (with the save's id), `direct` (a game master's line), `mic`
(muted or not), `settings` (the speech cooldown).

From mate: `hello`, `action` (and its `option`), `say` (the line as it starts), `heard` (what
mate heard you say), `view` (send the player's view), `stop_audio`, `audio_end`.

## Licence

MIT. The models, Breeze and llama.cpp have their own licences.
