## Gemma 2B IT Chatbot (Gradio)

A simple Gradio chatbot powered by a Hugging Face Transformers text-generation pipeline. By default it uses `google/gemma-2b-it` and replies in Traditional Chinese.

### Features
- Uses `Transformers` pipeline for chat-style text generation
- Gradio UI with sliders for `max_new_tokens`, `temperature`, and `top_p`
- Auto-detects CUDA; falls back to CPU when no GPU is available
- Configurable via environment variables

### Setup & Run Service
#### Install required packages
```shell
pip install -r requirements.txt
```
#### Hugging Face access
```shell
huggingface-cli login
```
#### Run Service
```shell
python ${fileName}
```

Open `http://localhost:7860` in your browser.

### Environment variables
- `HF_MODEL_ID`: Override the model id. Default: `google/gemma-2b-it`.
- `SYSTEM_PROMPT`: Customize the system prompt. Default instructs replies in Traditional Chinese.
- `PORT`: Server port. Default: `7860`.

### Parameter descriptions
- `max_new_tokens`: Maximum number of new tokens generated per response. Higher means longer outputs but slower generation.
- `temperature`: Sampling temperature (randomness). Lower is more deterministic; higher is more creative but can drift off-topic.
- `top_p`: Nucleus sampling threshold. Sample only from the smallest set of tokens whose cumulative probability ≥ p; lower is more conservative.
