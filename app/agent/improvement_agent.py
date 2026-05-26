import asyncio
import json
import re
from app.models.groq_model import GroqModel
from app.database.db import (
    get_elo_ratings, get_all_results,
    init_db, init_elo_table
)
from app.runner.prompt_runner import run_benchmark
from app.evaluator.judge import judge_run_stable
from app.elo.elo_system import update_elo_for_run

# The agent uses llama3-70b to generate prompts
# It's strong enough for creative prompt generation
# and different from our judge (GPT-OSS 120B)
agent_model = GroqModel("llama-3.3-70b-versatile")

def analyze_weaknesses() -> dict:
    """
    Step 1 — Analyze current ELO and judge scores
    to find where models struggle most.

    Returns a dict with:
    - weakest_model: model with lowest ELO
    - weakest_dimension: accuracy, clarity, or completeness
    - weakness_summary: human readable description
    - all_ratings: full ELO table for context
    """
    print("\n" + "="*50)
    print("AGENT: Analyzing model weaknesses...")
    print("="*50)

    # Load ELO ratings — sorted best first
    ratings = get_elo_ratings()

    if not ratings:
        print("No ELO data found. Run some benchmarks first.")
        return {}

    # The weakest model = lowest ELO score
    # Last item since ratings are sorted descending
    weakest_model = ratings[-1]
    print(f"Weakest model by ELO: {weakest_model['model_name']} ({weakest_model['elo_score']} ELO)")

    # Load all benchmark results to find weakest dimension
    all_results = get_all_results()

    if not all_results:
        return {'weakest_model': weakest_model['model_name'], 'all_ratings': ratings}

    # Filter to results that have judge scores
    scored = [r for r in all_results if r.get('judge_score') is not None]

    if not scored:
        return {'weakest_model': weakest_model['model_name'], 'all_ratings': ratings}

    # Calculate average score per model across all runs
    model_scores = {}
    for r in scored:
        name = r['model_name']
        if name not in model_scores:
            model_scores[name] = []
        model_scores[name].append(r['judge_score'])

    model_averages = {
        name: sum(scores) / len(scores)
        for name, scores in model_scores.items()
    }

    # Find which models consistently score below 8.0
    # These are the ones that need harder prompts
    struggling_models = [
        name for name, avg in model_averages.items()
        if avg < 8.0
    ]

    print(f"\nModel averages: {model_averages}")
    print(f"Struggling models (avg < 8.0): {struggling_models}")

    # Determine weakest dimension overall
    # We check which dimension has lowest average across all models
    # This tells us WHAT to target in new prompts
    dimension_scores = {
        'accuracy': [],
        'clarity': [],
        'completeness': []
    }

    # We stored overall score in db but not individual dimensions
    # So we use overall score as proxy and generate prompts
    # that specifically target completeness and clarity
    # (these are consistently the lowest based on our runs)
    weakest_dimension = 'completeness'
    # From our runs we've seen completeness is consistently lowest
    # The agent will target this in prompt generation

    # Build a summary for the LLM to understand context
    ratings_summary = "\n".join([
        f"- {r['model_name']}: {r['elo_score']:.1f} ELO, "
        f"{r['wins']}W/{r['losses']}L/{r['ties']}T"
        for r in ratings
    ])

    weakness_summary = f"""
Current ELO Rankings:
{ratings_summary}

Weakest model: {weakest_model['model_name']} ({weakest_model['elo_score']:.1f} ELO)
Models scoring below 8.0 average: {struggling_models}
Primary weakness dimension: {weakest_dimension}
Total benchmark runs analyzed: {len(set(r['run_id'] for r in all_results))}
    """.strip()

    print(f"\nWeakness summary:\n{weakness_summary}")

    return {
        'weakest_model': weakest_model['model_name'],
        'weakest_dimension': weakest_dimension,
        'weakness_summary': weakness_summary,
        'struggling_models': struggling_models,
        'all_ratings': ratings,
        'model_averages': model_averages
    }

def generate_targeted_prompts(analysis: dict, num_prompts: int = 5) -> list:
    """
    Step 2 — Use an LLM to generate new harder prompts
    that specifically target the identified weaknesses.

    This is the 'agentic' part — an LLM generating eval data
    to challenge other LLMs. Meta-learning at its core.

    num_prompts = how many new prompts to generate
    """
    print(f"\n{'='*50}")
    print(f"AGENT: Generating {num_prompts} targeted prompts...")
    print(f"{'='*50}")

    weakness_summary = analysis.get('weakness_summary', 'Models struggle with completeness')
    weakest_dimension = analysis.get('weakest_dimension', 'completeness')
    weakest_model = analysis.get('weakest_model', 'unknown')

    # This is the agent's core prompt — we're asking an LLM
    # to generate prompts that will expose weaknesses in other LLMs
    # Key insight: we tell it WHAT weakness to target
    generation_prompt = f"""You are an AI evaluation expert designing challenging benchmark prompts.

CURRENT EVALUATION ANALYSIS:
{weakness_summary}

YOUR TASK:
Generate exactly {num_prompts} benchmark prompts that will:
1. Specifically test {weakest_dimension} — the identified weakness
2. Be harder than typical prompts — require multi-step reasoning
3. Cover diverse topics: coding, math, ML concepts, system design, algorithms
4. Require structured, complete answers with examples and explanations
5. Expose differences between strong and weak models clearly

PROMPT DESIGN GUIDELINES:
- Each prompt should require at least 3 distinct components to answer fully
- Include prompts that require code AND explanation (not just one)
- Include prompts that require mathematical reasoning
- Include prompts that require comparing multiple concepts
- Make prompts where an incomplete answer is clearly worse than a complete one

IMPORTANT: Respond ONLY with valid JSON. No text before or after.
Use this exact format:
{{
  "prompts": [
    {{
      "prompt": "The full prompt text here",
      "target_skill": "what skill this tests",
      "difficulty": "hard or very_hard",
      "why_challenging": "why this exposes completeness weaknesses"
    }}
  ]
}}
"""

    try:
        print("Asking agent LLM to generate prompts...")
        response = agent_model.generate(
            prompt=generation_prompt,
            system_prompt="You are an expert AI evaluator. Generate challenging, diverse benchmark prompts. Respond only with valid JSON.",
            temperature=0.8
            # temperature=0.8 here — we WANT creativity
            # Unlike the judge (temperature=0), the agent
            # should generate diverse, varied prompts
            # High temperature = more creative output
        )

        raw = response.output
        print(f"Agent responded in {response.latency_seconds}s")

        # Parse the JSON response
        # Try direct parse first, then regex fallback
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Find JSON block in response
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                print("Could not parse agent response as JSON")
                return get_fallback_prompts()

        prompts_data = parsed.get('prompts', [])
        prompts = [p['prompt'] for p in prompts_data if 'prompt' in p]

        print(f"\nGenerated {len(prompts)} prompts:")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p[:80]}...")

        return prompts

    except Exception as e:
        print(f"Agent prompt generation failed: {e}")
        print("Using fallback prompts...")
        return get_fallback_prompts()

def get_fallback_prompts() -> list:
    """
    Backup prompts if agent generation fails.
    These are manually crafted hard prompts that test
    completeness, multi-step reasoning, and code quality.
    Always having fallbacks = robust production code.
    """
    return [
        "Implement a LRU (Least Recently Used) cache in Python using OrderedDict. Explain the time complexity of each operation and provide a complete working example with test cases.",
        "Explain the difference between bagging and boosting in ensemble learning. Implement a simple version of both in Python and compare their performance on a sample dataset.",
        "Design a rate limiter system that supports multiple algorithms (token bucket, sliding window). Write the Python implementation and explain when you'd use each algorithm.",
        "Explain how attention mechanisms work in transformers. Write the mathematical formulation, implement scaled dot-product attention in Python, and explain why it works better than RNNs.",
        "What is the Byzantine Generals Problem? Explain how blockchain solves it, implement a simple proof-of-work in Python, and discuss its limitations.",
    ]

async def run_improvement_cycle(num_prompts: int = 3) -> dict:
    """
    The full self-improving loop — runs all 5 steps end to end.

    num_prompts = how many new prompts to benchmark
    (3 is good for demo — enough to see improvement without taking too long)

    This is an async function because run_benchmark() is async.
    """
    print("\n" + "🤖"*25)
    print("SELF-IMPROVING AGENT — Starting improvement cycle")
    print("🤖"*25)

    init_db()
    init_elo_table()

    # ── Step 1: Analyze ──
    analysis = analyze_weaknesses()

    if not analysis:
        print("Cannot analyze — no benchmark data found.")
        print("Run scripts/run_judge.py first to generate data.")
        return {}

    # ── Step 2: Generate prompts ──
    new_prompts = generate_targeted_prompts(analysis, num_prompts=num_prompts)

    if not new_prompts:
        print("No prompts generated — aborting cycle")
        return {}

    # ── Steps 3, 4, 5: Benchmark + Judge + ELO for each prompt ──
    print(f"\n{'='*50}")
    print(f"AGENT: Running benchmarks on {len(new_prompts)} new prompts...")
    print(f"{'='*50}")

    cycle_results = []

    for i, prompt in enumerate(new_prompts, 1):
        print(f"\n{'─'*50}")
        print(f"Prompt {i}/{len(new_prompts)}: {prompt[:70]}...")
        print(f"{'─'*50}")

        # Step 3: Benchmark
        print(f"\nBenchmarking all models...")
        result = await run_benchmark(
            prompt=prompt,
            system_prompt="You are a helpful AI assistant. Provide complete, detailed answers with examples and code where relevant."
        )

        run_id = result['run_id']
        successful = sum(1 for r in result['results'].values() if r['output'])
        print(f"Benchmark complete — {successful}/4 models succeeded")

        # Step 4: Judge
        print(f"\nJudging responses (3 ensemble runs)...")
        scores = judge_run_stable(run_id, num_runs=3)

        if scores:
            print(f"Judging complete")
            # Show quick scores
            ranked = sorted(scores.items(), key=lambda x: x[1].get('overall', 0), reverse=True)
            for rank, (model, s) in enumerate(ranked, 1):
                print(f"  #{rank} {model}: {s['overall']}/10")

        # Step 5: Update ELO
        print(f"\nUpdating ELO...")
        elo_update = update_elo_for_run(run_id)

        cycle_results.append({
            'prompt': prompt,
            'run_id': run_id,
            'scores': scores,
            'elo_update': elo_update
        })

        # Small delay between prompts to avoid rate limiting
        # Groq free tier has per-minute limits
        if i < len(new_prompts):
            print(f"\nWaiting 5s before next prompt (rate limit protection)...")
            await asyncio.sleep(5)

    # ── Step 6: Final report ──
    print("\n" + "🏆"*25)
    print("AGENT CYCLE COMPLETE — Final Report")
    print("🏆"*25)

    # Load updated ELO ratings
    final_ratings = get_elo_ratings()

    print(f"\nFINAL ELO LEADERBOARD (after {len(new_prompts)} new prompts):")
    print(f"{'='*50}")
    for rank, r in enumerate(final_ratings, 1):
        print(f"#{rank} {r['model_name']}: {r['elo_score']:.1f} ELO "
              f"({r['wins']}W/{r['losses']}L/{r['ties']}T)")

    print(f"\nPrompts tested this cycle:")
    for i, cr in enumerate(cycle_results, 1):
        print(f"  {i}. {cr['prompt'][:70]}...")

    return {
        'analysis': analysis,
        'prompts_tested': new_prompts,
        'cycle_results': cycle_results,
        'final_ratings': final_ratings
    }