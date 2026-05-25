import json
import re
from app.models.groq_model import GroqModel
from app.database.db import get_results_by_run, update_judge_scores

# Independent judge — completely different organization from all 4 contestants
# openai/gpt-oss-120b = 120B params, different org, zero conflict of interest
# temperature=0 is set at call time for consistency
judge_model = GroqModel("openai/gpt-oss-120b")

def build_judge_prompt(original_prompt: str, responses: list) -> str:
    """
    Builds the prompt we send to the judge LLM.
    The quality of your judge prompt determines the quality of your scores.

    We give the judge:
    1. The original question
    2. All model responses labeled by model name
    3. Very specific instructions on what to score and how
    4. A strict JSON format to respond in
    """
    responses_text = ""
    for r in responses:
        # Strip <think> blocks from qwen3-32b — they hurt clarity scores unfairly
        # The thinking process is internal and shouldn't be judged as output
        output = r['output']
        if '<think>' in output:
            # Remove everything between <think> and </think> tags
            output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()

        responses_text += f"\nModel: {r['model_name']}\n"
        responses_text += f"Response: {output[:1500]}\n"
        # [:1500] = increased from 1000 to give models more room
        # since we now have max_tokens=2048 giving longer responses
        responses_text += "-" * 40 + "\n"

    prompt = f"""You are an expert AI evaluator. Your job is to score the quality of AI model responses.

ORIGINAL QUESTION:
{original_prompt}

MODEL RESPONSES TO EVALUATE:
{responses_text}

SCORING INSTRUCTIONS:
Score each model response on these 3 dimensions (1-10 scale):
- accuracy: Is the information correct and factually sound? (10 = perfect, 1 = completely wrong)
- clarity: Is the response clear, well-structured and easy to understand? (10 = crystal clear, 1 = confusing)
- completeness: Does it fully answer the question with good examples? (10 = complete, 1 = barely answers)

Then give an overall score (average of the 3) and a brief reasoning (1-2 sentences max).

IMPORTANT RULES:
- Be consistent and objective
- Do not favor any model based on name or length alone
- Judge only the content quality
- overall = (accuracy + clarity + completeness) / 3, rounded to 2 decimal places

Respond ONLY with valid JSON. No text before or after. Use this exact format:
{{
  "scores": {{
    "model_name_here": {{
      "accuracy": 8,
      "clarity": 7,
      "completeness": 9,
      "overall": 8.0,
      "reasoning": "Brief explanation here"
    }}
  }}
}}

Replace model_name_here with the actual model names from the responses above.
"""
    return prompt

def parse_judge_response(judge_output: str) -> dict:
    """
    Extracts the JSON scores from the judge's response.
    Has 3 fallback strategies to handle different judge output formats.
    """
    # Strategy 1: output is pure JSON
    try:
        return json.loads(judge_output)
    except json.JSONDecodeError:
        pass

    # Strategy 2: JSON is wrapped in markdown code block ```json ... ```
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', judge_output, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find any { } block in the text
    json_match = re.search(r'\{.*\}', judge_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    print(f"Warning: Could not parse judge response as JSON")
    print(f"Raw response: {judge_output[:500]}")
    return {}

def judge_run(run_id: str) -> dict:
    """
    Scores all model responses for one benchmark run.

    Steps:
    1. Load all responses for this run from SQLite
    2. Strip think blocks from reasoning models
    3. Build the judge prompt
    4. Send to judge LLM at temperature=0 (deterministic)
    5. Parse JSON scores
    6. Save scores back to SQLite
    7. Return results
    """
    print(f"\n{'='*50}")
    print(f"Judge evaluating run: {run_id}")
    print(f"{'='*50}")

    # Step 1: Load all responses for this run
    results = get_results_by_run(run_id)

    if not results:
        print(f"No results found for run_id: {run_id}")
        return {}

    original_prompt = results[0]['prompt']
    print(f"Prompt: {original_prompt[:80]}...")
    print(f"Models to judge: {[r['model_name'] for r in results]}")

    # Step 2: Build judge prompt
    judge_prompt = build_judge_prompt(original_prompt, results)

    # Step 3: Call judge at temperature=0
    # temperature=0 = fully deterministic — same input always gives same output
    # This is critical for consistent scores across multiple runs
    print(f"\nSending to judge LLM (temperature=0 for consistency)...")
    try:
        judge_response = judge_model.generate(
            prompt=judge_prompt,
            system_prompt="You are an expert AI evaluator. Always respond with valid JSON only. Be consistent and objective.",
            temperature=0.0
        )
        raw_output = judge_response.output
        print(f"Judge responded in {judge_response.latency_seconds}s")
    except Exception as e:
        print(f"Judge LLM failed: {e}")
        return {}

    # Step 4: Parse scores
    parsed = parse_judge_response(raw_output)

    if not parsed or 'scores' not in parsed:
        print("Could not extract scores from judge response")
        return {}

    scores = parsed['scores']

    # Step 5: Save and display
    print(f"\n{'='*50}")
    print(f"JUDGE SCORES — Run: {run_id}")
    print(f"{'='*50}")

    final_results = {}

    for result in results:
        model_name = result['model_name']

        # Fuzzy match model name — judge sometimes shortens names
        model_scores = None
        for key in scores:
            if model_name.lower() in key.lower() or key.lower() in model_name.lower():
                model_scores = scores[key]
                break

        if model_scores:
            # Recalculate overall ourselves to avoid judge math errors
            acc = model_scores.get('accuracy', 0)
            cla = model_scores.get('clarity', 0)
            com = model_scores.get('completeness', 0)
            overall = round((acc + cla + com) / 3, 2)
            # We recalculate instead of trusting judge's overall
            # because judges sometimes make arithmetic mistakes

            reasoning = model_scores.get('reasoning', 'No reasoning provided')

            update_judge_scores(
                result_id=result['id'],
                score=overall,
                reasoning=reasoning
            )

            print(f"\nModel: {model_name}")
            print(f"  Accuracy    : {acc}/10")
            print(f"  Clarity     : {cla}/10")
            print(f"  Completeness: {com}/10")
            print(f"  Overall     : {overall}/10")
            print(f"  Reasoning   : {reasoning}")

            model_scores['overall'] = overall
            final_results[model_name] = model_scores
        else:
            print(f"\nModel: {model_name} — no scores found in judge output")

    print(f"\n{'='*50}")
    return final_results

def judge_run_stable(run_id: str, num_runs: int = 3) -> dict:
    """
    Runs the judge multiple times and averages scores.
    Called 'ensemble judging' — used in real research papers like MT-Bench.

    Even at temperature=0, minor internal model variations can occur.
    3 runs + averaging gives much more stable, trustworthy results.

    num_runs=3 is the sweet spot:
    - 1 run = fast but potentially noisy
    - 3 runs = stable without too many API calls
    - 5+ runs = diminishing returns
    """
    print(f"\nRunning ensemble judge ({num_runs} runs)...")
    all_scores = []

    for i in range(num_runs):
        print(f"\n--- Judge pass {i+1}/{num_runs} ---")
        scores = judge_run(run_id)
        if scores:
            all_scores.append(scores)

    if not all_scores:
        print("All judge runs failed")
        return {}

    # Average scores across all successful runs
    final_scores = {}
    # Use models from first successful run as reference
    models = all_scores[0].keys()

    for model in models:
        final_scores[model] = {}

        for metric in ['accuracy', 'clarity', 'completeness']:
            # Collect this metric's value from every run
            values = [
                run[model].get(metric, 0)
                for run in all_scores
                if model in run
            ]
            # Average them — this smooths out single-run outliers
            final_scores[model][metric] = round(sum(values) / len(values), 2)

        # Recalculate overall from averaged dimensions
        acc = final_scores[model]['accuracy']
        cla = final_scores[model]['clarity']
        com = final_scores[model]['completeness']
        final_scores[model]['overall'] = round((acc + cla + com) / 3, 2)

        # Use reasoning from the last run — most recent is most reliable
        final_scores[model]['reasoning'] = all_scores[-1][model].get('reasoning', '')

    # Save final averaged scores back to database
    results = get_results_by_run(run_id)
    for result in results:
        model_name = result['model_name']
        if model_name in final_scores:
            update_judge_scores(
                result_id=result['id'],
                score=final_scores[model_name]['overall'],
                reasoning=final_scores[model_name]['reasoning']
            )

    print(f"\n{'='*50}")
    print(f"STABLE SCORES — averaged over {len(all_scores)} judge runs")
    print(f"{'='*50}")

    ranked = sorted(
        final_scores.items(),
        key=lambda x: x[1].get('overall', 0),
        reverse=True
    )
    for rank, (model, scores) in enumerate(ranked, 1):
        print(f"#{rank} {model}: {scores['overall']}/10  "
              f"(A:{scores['accuracy']} C:{scores['clarity']} Co:{scores['completeness']})")

    return final_scores