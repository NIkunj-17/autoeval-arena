import itertools
from app.database.db import (
    get_elo_ratings, upsert_elo,
    save_elo_match, get_results_by_run
)

# K factor = how much ELO changes per match
# 32 = standard value used in chess for newer players
# Higher K = ratings change faster (more volatile)
# Lower K = ratings change slower (more stable)
# 32 is the sweet spot for a benchmark platform
K_FACTOR = 32

# Starting ELO for every new model
INITIAL_ELO = 1000.0

def get_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculates the EXPECTED probability of model A beating model B.
    
    This is the core ELO formula — based on the logistic function.
    
    Examples:
    - Both models at 1000 ELO → expected = 0.5 (50/50 chance)
    - Model A at 1200, B at 1000 → expected ≈ 0.76 (A favored heavily)
    - Model A at 800, B at 1000 → expected ≈ 0.24 (A is underdog)
    
    The 400 divisor is the standard ELO constant.
    It means a 400-point difference = ~10x more likely to win.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def calculate_elo_change(rating: float, expected: float, actual: float) -> float:
    """
    Calculates how many ELO points a model gains or loses.
    
    actual_score:
    - 1.0 = won the match
    - 0.5 = tied the match  
    - 0.0 = lost the match
    
    If expected = 0.5 (fair match) and you win (actual=1.0):
    change = 32 * (1.0 - 0.5) = +16 points
    
    If expected = 0.9 (you were heavily favored) and you win:
    change = 32 * (1.0 - 0.9) = +3.2 points (small reward, expected)
    
    If expected = 0.9 and you LOSE:
    change = 32 * (0.0 - 0.9) = -28.8 points (big penalty, upset)
    """
    return K_FACTOR * (actual - expected)

def get_current_ratings() -> dict:
    """
    Loads current ELO ratings from database into a dict.
    If a model doesn't exist yet, gives it INITIAL_ELO.
    
    Returns: {'llama3-8b': 1000.0, 'llama3-70b': 1024.5, ...}
    """
    ratings_list = get_elo_ratings()
    # Convert list of dicts to simple name→score dict
    ratings = {r['model_name']: r['elo_score'] for r in ratings_list}
    return ratings

def get_current_stats() -> dict:
    """
    Loads wins/losses/ties for each model from database.
    Returns: {'llama3-8b': {'wins': 5, 'losses': 3, 'ties': 1, 'total': 9}}
    """
    ratings_list = get_elo_ratings()
    stats = {}
    for r in ratings_list:
        stats[r['model_name']] = {
            'wins': r['wins'],
            'losses': r['losses'],
            'ties': r['ties'],
            'total_matches': r['total_matches']
        }
    return stats

def determine_winner(score_a: float, score_b: float) -> str:
    """
    Determines who won a head-to-head based on judge scores.
    
    Returns:
    - 'a' if model A won
    - 'b' if model B won
    - 'tie' if scores are within 0.5 points of each other
    
    Why 0.5 threshold for tie?
    Judge scores have natural variance — a difference less than 0.5
    is not statistically meaningful, so we call it a tie.
    """
    diff = abs(score_a - score_b)
    if diff < 0.5:
        return 'tie'
    elif score_a > score_b:
        return 'a'
    else:
        return 'b'

def update_elo_for_run(run_id: str) -> dict:
    """
    Main function — updates ELO ratings after a benchmark run.
    
    How it works:
    1. Load judge scores for this run from SQLite
    2. Generate all possible head-to-head pairs
       (4 models → 6 pairs: AB, AC, AD, BC, BD, CD)
    3. For each pair, determine winner based on judge score
    4. Calculate ELO changes using the formula
    5. Update ratings in database
    6. Save match history
    
    itertools.combinations() = generates all pairs without repetition
    e.g. combinations(['A','B','C','D'], 2) →
         [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')]
    """
    print(f"\n{'='*50}")
    print(f"Updating ELO for run: {run_id}")
    print(f"{'='*50}")

    # Step 1: Load results for this run
    results = get_results_by_run(run_id)

    # Filter to only models that have judge scores
    # Models without scores can't be compared fairly
    scored = [r for r in results if r.get('judge_score') is not None]

    if len(scored) < 2:
        print("Need at least 2 scored models to update ELO")
        return {}

    print(f"Models with scores: {[r['model_name'] for r in scored]}")

    # Step 2: Load current ratings and stats
    ratings = get_current_ratings()
    stats = get_current_stats()

    # Initialize any new models that haven't been seen before
    for r in scored:
        name = r['model_name']
        if name not in ratings:
            ratings[name] = INITIAL_ELO
            print(f"New model '{name}' initialized at ELO {INITIAL_ELO}")
        if name not in stats:
            stats[name] = {'wins': 0, 'losses': 0, 'ties': 0, 'total_matches': 0}

    # Step 3: Generate all head-to-head pairs
    # We make a dict for quick score lookup: {'llama3-8b': 8.5, ...}
    score_map = {r['model_name']: r['judge_score'] for r in scored}

    # itertools.combinations generates every unique pair
    pairs = list(itertools.combinations(scored, 2))
    print(f"Head-to-head matchups: {len(pairs)}")

    elo_changes = {r['model_name']: 0.0 for r in scored}
    # Track cumulative ELO changes across all matchups
    # A model might gain from one matchup and lose from another

    # Step 4: Process each pair
    for result_a, result_b in pairs:
        name_a = result_a['model_name']
        name_b = result_b['model_name']
        score_a = score_map[name_a]
        score_b = score_map[name_b]

        # Get current ratings for this pair
        elo_a = ratings[name_a]
        elo_b = ratings[name_b]

        # Calculate expected scores (what ELO predicts)
        expected_a = get_expected_score(elo_a, elo_b)
        expected_b = get_expected_score(elo_b, elo_a)
        # Note: expected_a + expected_b always = 1.0

        # Determine actual outcome
        outcome = determine_winner(score_a, score_b)

        if outcome == 'a':
            actual_a, actual_b = 1.0, 0.0
            winner_name = name_a
            stats[name_a]['wins'] += 1
            stats[name_b]['losses'] += 1
        elif outcome == 'b':
            actual_a, actual_b = 0.0, 1.0
            winner_name = name_b
            stats[name_b]['wins'] += 1
            stats[name_a]['losses'] += 1
        else:  # tie
            actual_a, actual_b = 0.5, 0.5
            winner_name = None
            stats[name_a]['ties'] += 1
            stats[name_b]['ties'] += 1

        stats[name_a]['total_matches'] += 1
        stats[name_b]['total_matches'] += 1

        # Calculate ELO changes
        change_a = calculate_elo_change(elo_a, expected_a, actual_a)
        change_b = calculate_elo_change(elo_b, expected_b, actual_b)

        # Accumulate changes — apply all at once after all matchups
        # This prevents order-of-processing from affecting results
        elo_changes[name_a] += change_a
        elo_changes[name_b] += change_b

        new_elo_a = ratings[name_a] + change_a
        new_elo_b = ratings[name_b] + change_b

        print(f"\n{name_a} ({score_a}/10) vs {name_b} ({score_b}/10)")
        print(f"  Winner: {winner_name or 'TIE'}")
        print(f"  ELO: {name_a} {elo_a:.1f} → {new_elo_a:.1f} ({change_a:+.1f})")
        print(f"  ELO: {name_b} {elo_b:.1f} → {new_elo_b:.1f} ({change_b:+.1f})")

        # Save match to history
        save_elo_match(
            run_id=run_id,
            model_a=name_a,
            model_b=name_b,
            winner=winner_name,
            elo_change_a=round(change_a, 2),
            elo_change_b=round(change_b, 2),
            new_elo_a=round(new_elo_a, 2),
            new_elo_b=round(new_elo_b, 2)
        )

    # Step 5: Apply all accumulated ELO changes
    # We do this AFTER all matchups so a model's rating mid-run
    # doesn't affect other matchups in the same run
    final_ratings = {}
    for name in elo_changes:
        ratings[name] += elo_changes[name]
        final_ratings[name] = round(ratings[name], 2)

        # Save updated rating to database
        upsert_elo(
            model_name=name,
            elo_score=final_ratings[name],
            wins=stats[name]['wins'],
            losses=stats[name]['losses'],
            ties=stats[name]['ties'],
            total_matches=stats[name]['total_matches']
        )

    # Step 6: Display final leaderboard
    print(f"\n{'='*50}")
    print("ELO LEADERBOARD (updated)")
    print(f"{'='*50}")
    sorted_ratings = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, elo) in enumerate(sorted_ratings, 1):
        change = elo_changes[name]
        print(f"#{rank} {name}: {elo} ELO  ({change:+.1f} this run)")

    return final_ratings