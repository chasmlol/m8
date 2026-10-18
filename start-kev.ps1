# Kev-4B decision server -> http://127.0.0.1:8080/v1/systemone
# Its KV cache at 8 bits (needs flash attention): half the VRAM of 16, so the game keeps its own.
& "$PSScriptRoot\llama.cpp\llama-server.exe" `
    -m "$PSScriptRoot\models\kev-4b\Kev-4B-Q8_0.gguf" `
    -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 `
    --cache-ram 0 `
    --host 127.0.0.1 --port 8080
