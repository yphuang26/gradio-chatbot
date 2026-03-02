## Gemma 3 27B IT Chatbot (Gradio)

A simple Gradio chatbot powered by **Google API** using the `gemma-3-27b-it` model.

此專案提供一個在瀏覽器中使用 Gemma 3 27B IT 的聊天機器人介面。

### Features

- Uses `google.generativeai` with the `gemma-3-27b-it` model
- Remembers conversation history and sends it to the model
- Simple Gradio `ChatInterface` UI

### Requirements

- Python 3.10+ (建議)
- A Google AI Studio API key (for Gemma 3)

### Setup & Run

#### 1. Install dependencies

```shell
pip install -r requirements.txt
```

#### 2. Configure API key

Edit `gemma3-chatbot.py` and replace the placeholder API key in:

```python
genai.configure(api_key="YOUR_API_KEY_HERE")
```

with your own Google AI Studio API key. https://aistudio.google.com/

#### 3. Run the app

```shell
python gemma3-chatbot.py
```

Then open `http://localhost:7860` in your browser to start chatting with Gemma 3 27B IT.

#### Reference

[Gemma 3 for Beginners: An Introduction to Google's Open-Source AI](https://huggingface.tw/blog/proflead/gemma-3-tutorial)
