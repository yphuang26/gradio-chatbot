import gradio as gr
from google import genai
from google.genai import types

client = genai.Client(api_key="[REPLACE-THIS-TEXT-WITH-YOUR-API-KEY]")


def respond(message: str, history: list[list[str]]):
    """
    Gradio ChatInference callback.
    - message: user's newest input message
    - history: [[user, bot], ...] conversation history
    """
    # Transform history to google.genai format
    contents: list[types.Content] = []
    for user_msg, bot_msg in history:
        if user_msg:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(user_msg)],
                )
            )
        if bot_msg:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(bot_msg)],
                )
            )

    # Add current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(message)],
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
)

if __name__ == "__main__":
    demo.launch()