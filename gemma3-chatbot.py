import gradio as gr
import google.generativeai as genai

genai.configure(api_key="AIzaSyDA4qOZa_GyTvmMFOR_LHW6vSOHVtZbxsM")
MODEL = genai.GenerativeModel("gemma-3-27b-it")


def respond(message: str, history: list[list[str]]):
    """
    Gradio ChatInference callback.
    - message: user's newest input message
    - history: [[user, bot], ...] conversation history
    """
    # Transform history to Google API format
    contents = []
    for user_msg, bot_msg in history:
        if user_msg:
            contents.append({"role": "user", "parts": [user_msg]})
        if bot_msg:
            contents.append({"role": "model", "parts": [bot_msg]})

    # Add current user message
    contents.append({"role": "user", "parts": [message]})

    response = MODEL.generate_content(contents=contents)
    reply = response.text
    return reply

demo = gr.ChatInterface(
    fn=respond,
    title="Gemma 3 27B IT Chatbot",
    description="A chatbot powered by Gemma 3 27B IT",
)

if __name__ == "__main__":
    demo.launch()