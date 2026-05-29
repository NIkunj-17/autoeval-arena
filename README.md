# 🏆 AutoEval Arena

### End-to-End LLM Evaluation & Benchmarking Platform

<p align="center">
  <a href="https://nikunj-17-autoeval-arena.hf.space">
    <img src="https://img.shields.io/badge/🔴_Live_Demo-HuggingFace-orange?style=for-the-badge">
  </a>
  <a href="https://github.com/NIkunj-17/autoeval-arena">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red">
  <img src="https://img.shields.io/badge/Groq-LLMs-green">
  <img src="https://img.shields.io/badge/License-MIT-purple">
</p>

---

## 🚀 Overview

**AutoEval Arena** is an automated benchmarking platform for evaluating and comparing Large Language Models (LLMs) using:

* ⚡ Parallel inference execution
* ⚖️ LLM-as-a-Judge scoring
* ♟️ ELO-based ranking system
* 🤖 Self-improving evaluation agents
* 📊 Real-time analytics dashboard

The platform enables developers and researchers to identify which model performs best for a given use case through reproducible, automated evaluations.

---

## 🔴 Live Demo

👉 https://nikunj-17-autoeval-arena.hf.space

---

# ✨ Features

| Feature                  | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| ⚡ Parallel Inference     | Executes multiple LLMs simultaneously using `asyncio.gather()`        |
| ⚖️ LLM-as-Judge          | GPT-OSS 120B evaluates outputs on accuracy, clarity, and completeness |
| ♟️ ELO Rankings          | Chess-style rating system inspired by LMSYS Chatbot Arena             |
| 🤖 Self-Improving Agent  | Automatically generates harder prompts based on weak model areas      |
| 📊 Interactive Dashboard | Leaderboards, radar charts, analytics, and score tracking             |
| 🆓 Zero Deployment Cost  | Built entirely using free-tier tools and APIs                         |

---

# 🏗️ System Architecture

```text
                     User Prompt
                          │
                          ▼
              Async Benchmark Runner
                 (asyncio.gather)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   llama-3.1-8b     qwen3-32b      llama-3.3-70b
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
                    SQLite Database
                          │
                          ▼
              Judge Model (GPT-OSS 120B)
           temperature=0 · 3-run ensemble
                          │
                          ▼
                 ELO Rating Calculation
                    6 pairwise matchups
                          │
                          ▼
                 Streamlit Dashboard
```

---

# 📊 Current Benchmark Rankings

| Rank | Model        | ELO  | Record       |
| ---- | ------------ | ---- | ------------ |
| 🥇   | llama3-70b   | 1034 | 5W / 2L / 8T |
| 🥈   | qwen3-32b    | 1002 | 6W / 4L / 5T |
| 🥉   | llama4-scout | 991  | 2W / 5L / 8T |
| 4️⃣  | llama3-8b    | 973  | 2W / 4L / 9T |

---

# 🛠️ Tech Stack

## Backend

* Python 3.12
* asyncio
* SQLite
* Groq SDK
* Google GenAI SDK

## Evaluation Engine

* LLM-as-Judge
* Ensemble Voting
* ELO Rating System
* Pairwise Matchups

## Frontend & Visualization

* Streamlit
* Plotly
* Pandas

## Agentic System

* Automated Prompt Generation
* Weakness Detection
* Feedback-Driven Re-Evaluation

## Deployment

* Docker
* HuggingFace Spaces

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/NIkunj-17/autoeval-arena
cd autoeval-arena
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure API Keys

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Get a free API key from:

👉 https://console.groq.com

---

# 🚀 Running the Project

## Run Benchmark Pipeline

```bash
python scripts/run_judge.py
```

This will:

* Generate model responses
* Run judge evaluation
* Store results in SQLite
* Update ELO ratings

---

## Launch Dashboard

```bash
streamlit run app/dashboard/app.py
```

---

## Run Self-Improving Agent

```bash
python scripts/run_agent.py
```

The agent will:

1. Analyze weakest models
2. Detect weak evaluation dimensions
3. Generate harder prompts
4. Re-run evaluations automatically

---

# 📁 Project Structure

```text
autoeval-arena/
│
├── app/
│   ├── models/          # LLM wrappers
│   ├── runner/          # Async benchmark execution
│   ├── evaluator/       # Judge evaluation logic
│   ├── elo/             # ELO rating calculations
│   ├── agent/           # Self-improving agent system
│   ├── database/        # SQLite operations
│   └── dashboard/       # Streamlit application
│
├── scripts/
│   ├── run_judge.py
│   └── run_agent.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🧠 Core Concepts

## ⚖️ LLM-as-a-Judge

Instead of relying on human annotators, the platform uses GPT-OSS 120B to evaluate responses based on:

* Accuracy
* Clarity
* Completeness
* Overall quality

This approach is inspired by:

* MT-Bench
* AlpacaEval
* LMSYS Arena

---

## 🗳️ Ensemble Judging

To reduce randomness and variance:

* Judge runs **3 independent evaluations**
* `temperature = 0`
* Scores are averaged

This improves evaluation consistency.

---

## ♟️ ELO Rating System

The platform uses a chess-inspired ELO ranking system:

```math
E = 1 / (1 + 10^((rB - rA)/400))
```

```math
new_rating = old_rating + K * (actual - expected)
```

Where:

* `K = 32`
* Models compete in pairwise matchups

---

## 🤖 Self-Improving Agent

The agent continuously improves benchmark quality by:

* Detecting weak-performing models
* Identifying weak evaluation dimensions
* Generating more difficult prompts
* Automatically re-running evaluations

This creates an autonomous evaluation loop.

---

# 📸 Dashboard Features

* 📈 ELO progression charts
* 🥇 Live leaderboard
* 📊 Radar comparison charts
* 📉 Win/Loss analytics
* 🧠 Judge scoring breakdowns

---

# 🐳 Docker Deployment

Build the container:

```bash
docker build -t autoeval-arena .
```

Run the container:

```bash
docker run -p 7860:7860 autoeval-arena
```

---

# 📌 Future Improvements

* Multi-judge consensus evaluation
* Human feedback integration
* Retrieval-Augmented Generation (RAG) benchmarks
* Long-context evaluation
* Code generation benchmarks
* Multi-modal evaluation support

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

### Nikunj Sharma

* 🌐 Portfolio: https://nikunj-portfolio-bice.vercel.app/
* 💻 GitHub: https://github.com/NIkunj-17

---

<p align="center">
Built in 7 days • Fully Open Source • Zero Cloud Cost
</p>
