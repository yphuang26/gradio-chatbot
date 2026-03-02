import gradio as gr
from google import genai
from google.genai import types

client = genai.Client(api_key="[REPLACE-THIS-TEXT-WITH-YOUR-API-KEY]")


def respond(message: str, history: list[dict]):
    """
    Gradio ChatInference callback.
    - message: user's newest input message
    - history: [[user, bot], ...] conversation history
    """
    # Transform history (messages format) to google.genai format
    contents: list[types.Content] = []
    for msg in history:
        role = msg.get("role")
        text = msg.get("content")
        if not text:
            continue

        # Map OpenAI-style roles to Gemini roles
        if role == "assistant":
            role = "model"

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=text)],
            )
        )

    # Add current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )
    )

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=contents,
    )

    # Prefer the high-level text helper when available
    reply = getattr(response, "text", None)
    if reply:
        return reply

    # Fallback: manually extract text from candidates
    if response and getattr(response, "candidates", None):
        candidate = response.candidates[0]
        if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
            parts = [
                part.text
                for part in candidate.content.parts
                if hasattr(part, "text") and part.text is not None
            ]
            if parts:
                return "".join(parts)

    return "No response generated."

demo = gr.ChatInterface(
    fn=respond,
    title="Gemma 3 27B IT Chatbot",
    description="A chatbot powered by Gemma 3 27B IT",
    type="messages",
)

if __name__ == "__main__":
    demo.launch()