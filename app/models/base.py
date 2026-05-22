from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ModelResponse:
    model_name: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float = 0.0

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> ModelResponse:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"