# Starts the mate's voice, Breeze TTS 2, in WSL on port 7860 (mate clones voices\soldier.wav with it).
# Only the per-frame stages run as CUDA graphs: as fast as --fast-all for the first audio, at about
# half the VRAM.
# (the model folder here, as WSL sees it)
$model = (wsl.exe -d Ubuntu -- wslpath -a "$($PSScriptRoot -replace '\\', '/')/models/breeze-tts-2").Trim()
wsl.exe -d Ubuntu -- bash -c "cd ~/breeze-tts && .venv/bin/python -m breeze_infer.api $model --host 0.0.0.0 --port 7860 --fast-backbone-decode --fast-depth-decoder --fast-codec"
