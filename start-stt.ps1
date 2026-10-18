# Speech to text: Parakeet TDT behind an OpenAI-style endpoint ->
# http://127.0.0.1:8083/v1/audio/transcriptions (m8's stt-server, built with m8).
& "$PSScriptRoot\m8\target\release\stt-server.exe" "$PSScriptRoot\models\sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8" 8083
