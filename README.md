# 🏆 AutoEval Arena

An end-to-end LLM evaluation platform that benchmarks multiple AI models using 
async parallel inference, LLM-as-judge ensemble scoring, ELO ratings, and a 
self-improving agent that auto-generates harder prompts targeting model weak spots.

## 🔴 Live Demo
👉 https://nikunj-17-autoeval-arena.hf.space

## Architecture
- 4 Models: Llama 3.1 8B, Llama 3.3 70B, Qwen3 32B, Llama4-Scout
- Judge: GPT-OSS 120B (independent org, temperature=0, 3-run ensemble)
- ELO System: Chess-style ratings — 6 head-to-head matchups per run
- Self-Improving Agent: Analyzes ELO weakness → generates targeted prompts → reruns
- Dashboard: Streamlit + Plotly

## Tech Stack
Python · Groq SDK · Streamlit · Plotly · SQLite · asyncio · HuggingFace Spaces

## Key Results
- llama3-70b: 1034 ELO (strongest on hard prompts)
- qwen3-32b: 1002 ELO (strong on easy, weak on complex reasoning)
- llama4-scout: 991 ELO
- llama3-8b: 973 ELO

## Setup
\```bash
git clone https://github.com/YOUR_USERNAME/autoeval-arena
cd autoeval-arena
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Add GROQ_API_KEY to .env
python scripts/run_judge.py      # run benchmark
python scripts/run_agent.py      # run self-improving agent
streamlit run app/dashboard/app.py  # launch dashboard
\```

## Concepts Demonstrated
- LLM-as-Judge with ensemble voting (reduces variance)
- ELO rating system (same algorithm as chess.com + LMSYS Chatbot Arena)
- Async parallel inference with asyncio.gather()
- Self-improving feedback loop (agentic AI)
- Judge independence (separate org from all contestants)
- Temperature=0 for deterministic scoring