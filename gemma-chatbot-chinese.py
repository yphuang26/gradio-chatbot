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
            max_new_tokens = gr.Slider(15, 256, value=128, step=1, label="max_new_tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top_p")

        chatbot = gr.Chatbot(label="聊天記錄", type="messages")
        msg = gr.Textbox(label="輸入訊息", placeholder="請輸入您的問題...")
        
        with gr.Row():
            submit_btn = gr.Button("送出", variant="primary")
            stop_btn = gr.Button("停止", variant="stop")
            clear = gr.Button("清除對話")

        def user(user_message, history, max_tokens, temp, top_p_val):
            return "", history + [{"role": "user", "content": user_message}]

        def bot(history, max_tokens, temp, top_p_val):
            if not history:
                return history
            
            user_message = history[-1]["content"]
            history_messages = history[:-1]  # Exclude the last user message
            
            # Generate response
            response = generate(user_message, history_messages, max_tokens, temp, top_p_val)
            history.append({"role": "assistant", "content": response})
            return history

        def stop_generation(history):
            if not history:
                return history, ""
            
            # Remove the last user message from chat history
            if history and history[-1]["role"] == "user":
                user_message = history[-1]["content"]
                history = history[:-1]  # Remove the last user message
                return history, user_message
            
            return history, ""

        # Send button event
        submit_btn.click(
            user, 
            [msg, chatbot, max_new_tokens, temperature, top_p], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot, 
            [chatbot, max_new_tokens, temperature, top_p], 
            chatbot
        )

        # Enter key event
        msg.submit(
            user, 
            [msg, chatbot, max_new_tokens, temperature, top_p], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot, 
            [chatbot, max_new_tokens, temperature, top_p], 
            chatbot
        )

        # Stop button event - stop generation and return user message to input
        stop_btn.click(
            stop_generation,
            [chatbot],
            [chatbot, msg]
        )
        
        # Clear chat button event
        clear.click(lambda: [], None, chatbot, queue=False)

    return demo


if __name__ == "__main__":
    demo = ui()
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, show_api=False)
