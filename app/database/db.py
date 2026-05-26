import sqlite3
import os
from datetime import datetime

# This is the path where our SQLite database file will be created
# It lives in the project root as autoeval.db
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "autoeval.db")

def get_connection():
    """
    Creates and returns a connection to the SQLite database.
    check_same_thread=False allows the connection to be used
    across different threads — important for async code later.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Row factory makes results return as dict-like objects
    # instead of plain tuples — so we can do row["model_name"]
    # instead of row[0]
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Creates the database tables if they don't exist yet.
    Called once at startup — safe to call multiple times.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # This table stores every single model response we ever get
    # Each row = one model's response to one prompt
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,          -- groups all models' responses for one prompt together
            prompt TEXT NOT NULL,          -- the question we asked
            model_name TEXT NOT NULL,      -- which model answered
            output TEXT NOT NULL,          -- what the model said
            input_tokens INTEGER,          -- tokens used in prompt
            output_tokens INTEGER,         -- tokens used in response
            latency_seconds REAL,          -- how long it took
            cost_usd REAL DEFAULT 0.0,     -- cost (0 for free models)
            judge_score REAL DEFAULT NULL, -- score given by judge LLM (Day 5)
            judge_reasoning TEXT DEFAULT NULL, -- why judge gave that score (Day 5)
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # This table stores high level info about each benchmark run
    # One run = one prompt sent to all models
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,           -- unique run ID (uuid)
            prompt TEXT NOT NULL,          -- the prompt used
            models_used TEXT NOT NULL,     -- comma separated list of models
            status TEXT DEFAULT 'pending', -- pending, completed, failed
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {DB_PATH}")

def save_result(run_id: str, prompt: str, model_name: str, output: str,
                input_tokens: int, output_tokens: int,
                latency_seconds: float, cost_usd: float = 0.0):
    """
    Saves one model's response to the database.
    Called once per model per benchmark run.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO benchmark_results
        (run_id, prompt, model_name, output, input_tokens, output_tokens, latency_seconds, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, prompt, model_name, output, input_tokens, output_tokens, latency_seconds, cost_usd))
    conn.commit()
    conn.close()

def save_run(run_id: str, prompt: str, models_used: list, status: str = "completed"):
    """
    Saves the high level run metadata.
    Called once per benchmark run after all models have responded.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO runs (id, prompt, models_used, status)
        VALUES (?, ?, ?, ?)
    """, (run_id, prompt, ",".join(models_used), status))
    conn.commit()
    conn.close()

def get_all_results():
    """
    Returns every benchmark result from the database.
    Used by the dashboard to display results.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM benchmark_results ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    # Convert Row objects to plain dicts for easy use
    return [dict(row) for row in rows]

def get_results_by_run(run_id: str):
    """
    Returns all model responses for one specific run.
    Used to compare models side by side for a single prompt.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM benchmark_results
        WHERE run_id = ?
        ORDER BY latency_seconds ASC
    """, (run_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_judge_scores(result_id: int, score: float, reasoning: str):
    """
    Updates an existing benchmark result row with judge scores.
    Called AFTER the judge LLM has scored a response.

    result_id = the specific row in benchmark_results to update
    score     = number from 1.0 to 10.0
    reasoning = why the judge gave that score
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE benchmark_results
        SET judge_score = ?, judge_reasoning = ?
        WHERE id = ?
    """, (score, reasoning, result_id))
    # UPDATE changes existing rows — unlike INSERT which adds new ones
    # SET says which columns to change
    # WHERE id = ? means only update THIS specific row, not all rows
    conn.commit()
    conn.close()

def get_results_by_run(run_id: str):
    """
    Returns all model responses for one specific run.
    We already wrote this — just confirming it exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM benchmark_results
        WHERE run_id = ?
        ORDER BY latency_seconds ASC
    """, (run_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def init_elo_table():
    """
    Creates the ELO ratings table.
    One row per model — stores their current ELO rating.
    Called once at startup alongside init_db().
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            model_name TEXT PRIMARY KEY,
            -- PRIMARY KEY = unique identifier, no duplicates allowed
            -- Each model has exactly ONE row in this table
            
            elo_score REAL DEFAULT 1000.0,
            -- Starting ELO = 1000, the universal standard baseline
            -- Above 1000 = better than average
            -- Below 1000 = worse than average
            
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            ties INTEGER DEFAULT 0,
            total_matches INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elo_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            -- Links back to the benchmark run that caused this ELO change
            
            model_a TEXT NOT NULL,
            model_b TEXT NOT NULL,
            -- The two models that faced each other
            
            winner TEXT,
            -- NULL = tie, otherwise = winning model name
            
            elo_change_a REAL,
            elo_change_b REAL,
            -- How many points each model gained/lost
            -- Positive = gained, Negative = lost
            
            new_elo_a REAL,
            new_elo_b REAL,
            -- Their ELO scores AFTER this match
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_elo_ratings():
    """
    Returns all model ELO ratings sorted best first.
    Used by dashboard to show the ELO leaderboard.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM elo_ratings
        ORDER BY elo_score DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_elo_history(limit: int = 50):
    """
    Returns recent ELO match history.
    limit = how many recent matches to return
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM elo_history
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def upsert_elo(model_name: str, elo_score: float,
               wins: int, losses: int, ties: int, total_matches: int):
    """
    INSERT or UPDATE a model's ELO rating.
    
    'Upsert' = UPDATE if exists, INSERT if not.
    SQLite does this with INSERT OR REPLACE.
    This means we never need to check if a model exists first.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO elo_ratings
        (model_name, elo_score, wins, losses, ties, total_matches, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (model_name, elo_score, wins, losses, ties, total_matches))
    conn.commit()
    conn.close()

def save_elo_match(run_id: str, model_a: str, model_b: str,
                   winner: str, elo_change_a: float, elo_change_b: float,
                   new_elo_a: float, new_elo_b: float):
    """
    Saves one head-to-head match result to history.
    Called once per model pair per benchmark run.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO elo_history
        (run_id, model_a, model_b, winner, elo_change_a, elo_change_b, new_elo_a, new_elo_b)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, model_a, model_b, winner, elo_change_a, elo_change_b, new_elo_a, new_elo_b))
    conn.commit()
    conn.close()