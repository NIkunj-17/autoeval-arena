import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.model_registry import get_model, get_all_models

prompt = "What is 2 + 2? Answer in one sentence."

print("=" * 50)
print("Testing all models...")
print("=" * 50)

for name, model in get_all_models().items():
    try:
        response = model.generate(prompt)
        print(f"\n✅ {name}")
        print(f"   Output   : {response.output.strip()}")
        print(f"   Latency  : {response.latency_seconds}s")
        print(f"   Tokens   : {response.input_tokens} in / {response.output_tokens} out")
    except Exception as e:
        print(f"\n❌ {name} failed: {e}")

print("\n" + "=" * 50)
print("Done!")