from app.llm.ollama_intent_client import OllamaIntentClient

schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "chat",
                "open_app",
                "close_window",
                "set_volume",
                "play_music",
                "pause_music",
                "shutdown_pc",
                "restart_pc",
                "sleep_mode",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "slots": {
            "type": "object",
            "additionalProperties": True,
        },
    },
    "required": ["intent", "confidence", "slots"],
}

client = OllamaIntentClient(model="hebe-intent")

result = client.chat_structured(
    system_prompt=(
        "You are a strict intent classifier and slot extractor for a local desktop assistant named Hebe. "
        "Return only valid JSON. "
        "Use only the allowed intents. "
        "If the user wants to open an app, use intent='open_app' and slot 'app_name'. "
        "If the user is just chatting, use intent='chat'. "
        "Do not invent intents."
    ),
    user_prompt=(
        'Classify this request and extract slots.\n'
        'User request: "abre obs"'
    ),
    schema=schema,
    temperature=0.0,
)

print(result)