from app.models.groq_model import GroqModel
from app.models.gemini_model import GeminiModel
from app.models.base import BaseModel

AVAILABLE_MODELS = {
    "llama3-8b": GroqModel("llama-3.1-8b-instant"),
    "llama3-70b": GroqModel("llama-3.3-70b-versatile"),
    "qwen3-32b": GroqModel("qwen/qwen3-32b"),
    "llama4-scout": GroqModel("meta-llama/llama-4-scout-17b-16e-instruct"),
}

def get_model(name: str) -> BaseModel:
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[name]

def get_all_models() -> dict:
    return AVAILABLE_MODELS