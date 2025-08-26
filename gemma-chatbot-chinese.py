import os
import torch
import gradio as gr
from typing import List, Tuple
from transformers import pipeline

MODEL_ID = os.environ.get("HF_MODEL_ID", "google/gemma-2b-it")
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Answer in Traditional Chinese (繁體中文)."
)

def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def load_pipeline():
    device = "cuda" if is_cuda_available() else "cpu"
    
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    return pipe


PIPE = load_pipeline()


def build_messages(messages: List[dict], user_message: str) -> List[dict]:
    # messages: list of {"role": "user"|"assistant", "content": str}
    chat_messages = []

    chat_messages.append({"role": "user", "content": DEFAULT_SYSTEM_PROMPT})
    chat_messages.append({"role": "assistant", "content": "我是一個有用的助手，我會用繁體中文回答您的問題。"})

    # Normalize and append history messages if provided in dict form
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            chat_messages.append({"role": role, "content": content})

    # Append current user message
    chat_messages.append({"role": "user", "content": user_message})
    return chat_messages


def generate(
    message,
    history: List[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    # Gradio ChatInterface with type="messages" provides message as dict and history as list[dict]
    if isinstance(message, dict):
        message_text = message.get("content", "")
    else:
        message_text = str(message)

    chat_messages = build_messages(history, message_text)
    
    # Using pipeline to generate response
    outputs = PIPE(
        chat_messages, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
        repetition_penalty=1.05,
    )
    
    # Extract the assistant's response
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response


def ui():
    with gr.Blocks(title="Gemma 2B IT Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        **Chatbot with Google Gemma 2B IT**  
        - 模型: `google/gemma-2b-it`  
        - 回答語言：預設繁體中文
        """)

        with gr.Row():
            max_new_tokens = gr.Slider(64, 2048, value=512, step=16, label="max_new_tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top_p")

        chat = gr.ChatInterface(
            fn=lambda message, history: generate(
                message,
                history,
                int(max_new_tokens.value),
                float(temperature.value),
                float(top_p.value),
            ),
            type="messages",
            multimodal=False,
            stop_btn="停止",
            submit_btn="送出",
        )

    return demo


if __name__ == "__main__":
    demo = ui()
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, show_api=False)
