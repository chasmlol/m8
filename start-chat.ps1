# Gemma 4 12B QAT, uncensored by HauhauCS (Balanced) -> http://127.0.0.1:8081/v1/chat/completions
# Its vision file (mmproj) lets it see the player's view (Kevin's eyes).
# Its KV cache at 8 bits: half the VRAM of 16, so the game keeps its own.
# Thinking is off so NPC lines come back immediately. Sampling per the model card; DRY stops it repeating itself.
& "$PSScriptRoot\llama.cpp\llama-server.exe" `
    -m "$PSScriptRoot\models\gemma-4-12b-hauhaucs\Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf" `
    --mmproj "$PSScriptRoot\models\gemma-4-12b-hauhaucs\mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf" `
    -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 `
    --cache-ram 512 `
    --reasoning off `
    --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.05 --dry-multiplier 0.8 `
    --host 127.0.0.1 --port 8081
