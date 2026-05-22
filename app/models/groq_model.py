import os
import time
from groq import Groq
from dotenv import load_dotenv
from app.models.base import BaseModel, ModelResponse

load_dotenv()

class GroqModel(BaseModel):
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        super().__init__(model_name)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, prompt: str, system_prompt: str = None) -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024
        )
        latency = round(time.time() - start, 3)

        return ModelResponse(
            model_name=self.model_name,
            output=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_seconds=latency,
            cost_usd=0.0
        )