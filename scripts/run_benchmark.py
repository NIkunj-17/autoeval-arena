import sys
import os
import asyncio

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.db import init_db
from app.runner.prompt_runner import run_benchmark, display_results

async def main():
    # Step 1 — Initialize database (creates tables if not exist)
    init_db()

    # Step 2 — Run a benchmark with a real eval prompt
    # This is a question where different models might give different answers
    # making it interesting to compare
    result = await run_benchmark(
        prompt="Explain the difference between supervised and unsupervised learning. Give one real world example of each.",
        system_prompt="You are a helpful AI assistant. Be concise and clear."
    )

    # Step 3 — Display results nicely in terminal
    display_results(result)

    # Step 4 — Run a second benchmark to show multiple runs work
    result2 = await run_benchmark(
        prompt="Write a Python function that checks if a string is a palindrome.",
        system_prompt="You are an expert Python developer. Write clean, commented code."
    )
    display_results(result2)

# asyncio.run() starts the async event loop and runs main()
# This is the standard way to run async code from a regular script
if __name__ == "__main__":
    asyncio.run(main())