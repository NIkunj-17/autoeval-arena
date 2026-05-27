import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.db import get_all_results, get_elo_ratings, get_elo_history, init_db, init_elo_table

init_db()
init_elo_table()

data = {
    "benchmark_results": get_all_results(),
    "elo_ratings": get_elo_ratings(),
    "elo_history": get_elo_history(limit=500)
}

# Save to data folder
os.makedirs("data", exist_ok=True)
with open("data/seed_data.json", "w") as f:
    json.dump(data, f, indent=2, default=str)

print(f"Exported {len(data['benchmark_results'])} benchmark results")
print(f"Exported {len(data['elo_ratings'])} ELO ratings")
print(f"Exported {len(data['elo_history'])} ELO history records")
print("Saved to data/seed_data.json")