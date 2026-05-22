import os
import time
from google import genai
from dotenv import load_dotenv
from app.models.base import BaseModel, ModelResponse

load_dotenv()

class GeminiModel(BaseModel):
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, system_prompt: str = None) -> ModelResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        start = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt
        )
        latency = round(time.time() - start, 3)
        output = response.text
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0

        return ModelResponse(
            model_name=self.model_name,
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            cost_usd=0.0
        )