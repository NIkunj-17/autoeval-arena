import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.db import init_db, init_elo_table
from app.runner.prompt_runner import run_benchmark, display_results
from app.evaluator.judge import judge_run
from app.elo.elo_system import update_elo_for_run
import asyncio

async def main():
    init_db()

    # Step 1 — Run a fresh benchmark
    # This sends the prompt to all 4 models and saves to SQLite
    print("Running benchmark...")
    result = await run_benchmark(
        prompt="What is the difference between a list and a tuple in Python? When should you use each?",
        system_prompt="You are a helpful Python expert. Be clear and concise."
    )
    display_results(result)

    # Step 2 — Judge the run
    # Takes the run_id from Step 1 and scores all responses
    run_id = result['run_id']
    print(f"\nNow judging run {run_id}...")
    judge_results = judge_run(run_id)

    # Step 3 — Show final ranked leaderboard
    if judge_results:
        print(f"\n{'='*50}")
        print("FINAL LEADERBOARD (sorted by score)")
        print(f"{'='*50}")

        # Sort models by overall score descending
        # .items() gives us (model_name, scores) pairs
        # sorted() with key=lambda sorts by overall score
        # reverse=True = highest score first
        ranked = sorted(
            judge_results.items(),
            key=lambda x: x[1].get('overall', 0),
            reverse=True
        )

        for rank, (model, scores) in enumerate(ranked, 1):
            print(f"#{rank} {model}: {scores.get('overall', '?')}/10")

from app.evaluator.judge import judge_run, judge_run_stable

async def main():
    init_db()
    
    result = await run_benchmark(
        prompt="What is the difference between a list and a tuple in Python? When should you use each?",
        system_prompt="You are a helpful Python expert. Be clear and concise."
    )
    display_results(result)
    
    run_id = result['run_id']
    print(f"\nRunning stable judge (3 runs averaged)...")
    
    # Use stable judging instead of single run
    final_scores = judge_run_stable(run_id, num_runs=3)

async def main():
    # Initialize both tables
    init_db()
    init_elo_table()

    result = await run_benchmark(
        prompt="What is the difference between a list and a tuple in Python? When should you use each?",
        system_prompt="You are a helpful Python expert. Be clear and concise."
    )
    display_results(result)

    run_id = result['run_id']

    print(f"\nRunning stable ensemble judge (3 runs averaged)...")
    judge_run_stable(run_id, num_runs=3)

    print(f"\nUpdating ELO ratings...")
    update_elo_for_run(run_id)

if __name__ == "__main__":
    asyncio.run(main())
    