import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from app.models.model_registry import get_all_models
from app.database.db import save_result, save_run

# ThreadPoolExecutor lets us run multiple blocking API calls
# at the same time in separate threads
# This is important because Groq SDK is synchronous (blocking)
# but we want to call all models simultaneously
executor = ThreadPoolExecutor(max_workers=10)

def _call_model(model_name: str, model, prompt: str, system_prompt: str = None):
    """
    Calls a single model and returns the response.
    This is a regular blocking function — runs inside a thread.
    The underscore prefix means it's private/internal.
    """
    try:
        print(f"  🚀 Calling {model_name}...")
        response = model.generate(prompt, system_prompt)
        print(f"  ✅ {model_name} done in {response.latency_seconds}s")
        return model_name, response, None  # name, response, error
    except Exception as e:
        print(f"  ❌ {model_name} failed: {e}")
        return model_name, None, str(e)  # name, no response, error message

async def run_benchmark(prompt: str, system_prompt: str = None) -> dict:
    """
    The main function — sends prompt to ALL models simultaneously.

    async def means this is an async function (coroutine).
    It can pause and resume, allowing other tasks to run meanwhile.
    This is what enables parallel execution.
    """
    # Generate a unique ID for this run
    # uuid4() creates a random unique string like "a1b2c3d4-..."
    run_id = str(uuid.uuid4())[:8]  # use first 8 chars for readability

    models = get_all_models()
    print(f"\n{'='*50}")
    print(f"🏁 Starting benchmark run: {run_id}")
    print(f"📝 Prompt: {prompt[:80]}...")
    print(f"🤖 Models: {list(models.keys())}")
    print(f"{'='*50}")

    # Get the current event loop — this manages all async operations
    loop = asyncio.get_event_loop()

    # Create a list of tasks — one per model
    # loop.run_in_executor() runs a blocking function in a thread
    # without blocking the main async loop
    # This is how we call all 4 models at the same time
    tasks = [
        loop.run_in_executor(executor, _call_model, name, model, prompt, system_prompt)
        for name, model in models.items()
    ]

    # asyncio.gather() runs ALL tasks simultaneously and waits
    # for ALL of them to finish before continuing
    # This is the key line that makes everything parallel
    results_raw = await asyncio.gather(*tasks)

    # Process results and save to database
    results = {}
    successful_models = []

    for model_name, response, error in results_raw:
        if response:
            # Save to database
            save_result(
                run_id=run_id,
                prompt=prompt,
                model_name=model_name,
                output=response.output,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_seconds=response.latency_seconds,
                cost_usd=response.cost_usd
            )
            results[model_name] = {
                "output": response.output,
                "latency": response.latency_seconds,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "error": None
            }
            successful_models.append(model_name)
        else:
            results[model_name] = {
                "output": None,
                "latency": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": error
            }

    # Save run metadata
    save_run(run_id, prompt, successful_models)

    print(f"\n{'='*50}")
    print(f"✅ Run {run_id} complete — {len(successful_models)}/{len(models)} models succeeded")
    print(f"{'='*50}\n")

    return {
        "run_id": run_id,
        "prompt": prompt,
        "results": results
    }

def display_results(benchmark_output: dict):
    """
    Pretty prints the benchmark results in the terminal.
    Sorts models by latency so fastest shows first.
    """
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS — Run ID: {benchmark_output['run_id']}")
    print(f"📝 Prompt: {benchmark_output['prompt'][:80]}")
    print(f"{'='*60}")

    # Sort by latency — fastest model first
    results = benchmark_output["results"]
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["latency"] or 999  # put failed models last
    )

    for rank, (model_name, data) in enumerate(sorted_results, 1):
        if data["error"]:
            print(f"\n#{rank} ❌ {model_name}")
            print(f"   Error: {data['error']}")
        else:
            print(f"\n#{rank} 🏆 {model_name} — {data['latency']}s")
            print(f"   Output : {data['output'][:200].strip()}")
            print(f"   Tokens : {data['input_tokens']} in / {data['output_tokens']} out")
    print(f"\n{'='*60}")