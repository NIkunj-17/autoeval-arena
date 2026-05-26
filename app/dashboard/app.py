import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.database.db import (
    init_db, get_all_results, get_results_by_run,
    get_elo_ratings, get_elo_history, init_elo_table
)
from app.runner.prompt_runner import run_benchmark
from app.evaluator.judge import judge_run_stable

st.set_page_config(
    page_title="AutoEval Arena",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .welcome-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .step-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        background: white;
    }
    .step-number {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
        margin-right: 10px;
    }
    .stat-box {
        background: #f8f9ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e8eaff;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    init_db()
    init_elo_table()
    results = get_all_results()
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)

def run_benchmark_sync(prompt, system_prompt=None):
    return asyncio.run(run_benchmark(prompt, system_prompt))

def get_leaderboard(df):
    if df.empty or 'judge_score' not in df.columns:
        return pd.DataFrame()
    leaderboard = df.dropna(subset=['judge_score']).groupby('model_name').agg(
        avg_score=('judge_score', 'mean'),
        avg_latency=('latency_seconds', 'mean'),
        total_runs=('run_id', 'nunique'),
        avg_output_tokens=('output_tokens', 'mean')
    ).reset_index()
    leaderboard['avg_score'] = leaderboard['avg_score'].round(2)
    leaderboard['avg_latency'] = leaderboard['avg_latency'].round(3)
    return leaderboard.sort_values('avg_score', ascending=False).reset_index(drop=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🏆 AutoEval Arena")
    st.caption("LLM Benchmarking Platform")
    st.divider()

    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "🏆 Leaderboard",
            "⚡ ELO Ratings",
            "🏁 Run Benchmark",
            "📊 Analytics",
            "📋 History"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # Quick stats in sidebar
    df_side = load_data()
    if not df_side.empty:
        total_runs = df_side['run_id'].nunique()
        total_models = df_side['model_name'].nunique()
        st.metric("Total runs", total_runs)
        st.metric("Models tracked", total_models)

    st.divider()
    st.caption("Judge: GPT-OSS 120B")
    st.caption("Models: Llama, Qwen, Scout")
    st.caption("[View on GitHub](https://github.com)")

df = load_data()

# ────────────────────────────────────────────
# PAGE: HOME
# ────────────────────────────────────────────
if page == "🏠 Home":

    # Welcome banner
    st.markdown("""
    <div class="welcome-box">
        <h1 style="margin:0; font-size:2rem;">🏆 AutoEval Arena</h1>
        <p style="margin:8px 0 0; font-size:1.1rem; opacity:0.9;">
            An open LLM benchmarking platform. We test AI models head-to-head
            on real prompts and rank them using a chess-style ELO system.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # What is this?
    st.subheader("What is this?")
    st.write("""
    AutoEval Arena automatically benchmarks multiple AI language models on the same prompt,
    scores their responses using an independent judge AI, and tracks performance over time
    using an ELO rating system — the same algorithm used in chess rankings and LMSYS Chatbot Arena.
    """)

    col1, col2, col3, col4 = st.columns(4)

    elo_data = get_elo_ratings()
    total_runs = df['run_id'].nunique() if not df.empty else 0
    total_models = df['model_name'].nunique() if not df.empty else 0
    total_matchups = len(get_elo_history(limit=1000)) if elo_data else 0

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size:2rem; font-weight:bold; color:#667eea">{total_runs}</div>
            <div style="color:gray; font-size:0.9rem;">Benchmark Runs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size:2rem; font-weight:bold; color:#667eea">{total_models}</div>
            <div style="color:gray; font-size:0.9rem;">Models Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size:2rem; font-weight:bold; color:#667eea">{total_matchups}</div>
            <div style="color:gray; font-size:0.9rem;">ELO Matchups</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        top_model = elo_data[0]['model_name'] if elo_data else "TBD"
        st.markdown(f"""
        <div class="stat-box">
            <div style="font-size:1.2rem; font-weight:bold; color:#667eea">{top_model}</div>
            <div style="color:gray; font-size:0.9rem;">Current #1 Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Current leaderboard snapshot
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Current ELO Rankings")
        if elo_data:
            medals = ["🥇", "🥈", "🥉", "4️⃣"]
            for i, r in enumerate(elo_data[:4]):
                st.markdown(
                    f"{medals[i]} **{r['model_name']}** — "
                    f"`{r['elo_score']:.0f} ELO` — "
                    f"{r['wins']}W / {r['losses']}L / {r['ties']}T"
                )
        else:
            st.info("No ELO data yet — run some benchmarks!")

    with col2:
        st.subheader("🤖 How it works")
        st.markdown("""
        1. **Prompt** — same question sent to all 4 models
        2. **Race** — all models answer simultaneously
        3. **Judge** — AI judge scores accuracy, clarity, completeness
        4. **ELO** — models gain/lose points based on head-to-head results
        5. **Agent** — finds weak spots, generates harder questions automatically
        """)

    st.divider()

    # How to use guide
    st.subheader("🚀 How to use this platform")

    st.markdown("""
    <div class="step-card">
        <span class="step-number">1</span>
        <strong>View the Leaderboard</strong> — see which model is currently winning
        across all benchmark runs. Updated after every test.
    </div>
    <div class="step-card">
        <span class="step-number">2</span>
        <strong>Run a Benchmark</strong> — type any question and watch all 4 models
        race to answer it. The AI judge scores them automatically.
    </div>
    <div class="step-card">
        <span class="step-number">3</span>
        <strong>Check ELO Ratings</strong> — see the chess-style rankings that track
        each model's performance over time across many different prompts.
    </div>
    <div class="step-card">
        <span class="step-number">4</span>
        <strong>Run the Agent</strong> — let the self-improving AI automatically
        generate harder prompts targeting model weak spots and retest everything.
    </div>
    <div class="step-card">
        <span class="step-number">5</span>
        <strong>Explore Analytics</strong> — dive into score distributions, latency
        vs quality charts, and how models improve over time.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Sample prompts to try
    st.subheader("💡 Try these prompts")
    sample_prompts = [
        "Explain how gradient descent works with a Python example",
        "What is the difference between SQL and NoSQL databases?",
        "Write a binary search algorithm and explain its time complexity",
        "Explain the CAP theorem in distributed systems",
        "What is the transformer architecture in deep learning?",
    ]

    cols = st.columns(2)
    for i, prompt in enumerate(sample_prompts):
        with cols[i % 2]:
            if st.button(f"▶ {prompt[:45]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state['quick_prompt'] = prompt
                st.info(f"Go to **🏁 Run Benchmark** in the sidebar to run this prompt!")

# ────────────────────────────────────────────
# PAGE: LEADERBOARD
# ────────────────────────────────────────────
elif page == "🏆 Leaderboard":
    st.title("🏆 Leaderboard")
    st.caption("Average judge scores across all benchmark runs — updated in real time")

    if df.empty:
        st.info("👋 No benchmark data yet. Go to **🏁 Run Benchmark** to get started!")
    else:
        leaderboard = get_leaderboard(df)

        if leaderboard.empty:
            st.info("Benchmarks exist but no judge scores yet. Run a benchmark with judging enabled.")
        else:
            # Podium
            medals = ["🥇", "🥈", "🥉"]
            colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
            cols = st.columns(min(3, len(leaderboard)))

            for i, col in enumerate(cols):
                if i < len(leaderboard):
                    row = leaderboard.iloc[i]
                    with col:
                        st.markdown(f"""
                        <div style='text-align:center; padding:1.25rem;
                             border:2px solid {colors[i]}; border-radius:14px;'>
                            <div style='font-size:2.5rem'>{medals[i]}</div>
                            <div style='font-size:1rem; font-weight:bold; margin:6px 0'>
                                {row['model_name']}
                            </div>
                            <div style='font-size:1.8rem; color:{colors[i]}; font-weight:bold'>
                                {row['avg_score']}/10
                            </div>
                            <div style='color:gray; font-size:0.8rem; margin-top:4px'>
                                ⚡ {row['avg_latency']}s avg
                            </div>
                            <div style='color:gray; font-size:0.8rem'>
                                📊 {row['total_runs']} runs
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Score Comparison")
                fig = px.bar(
                    leaderboard, x='model_name', y='avg_score',
                    color='model_name',
                    labels={'model_name': 'Model', 'avg_score': 'Avg Score'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(
                    showlegend=False, yaxis_range=[0, 10],
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Speed Comparison")
                fig2 = px.bar(
                    leaderboard, x='model_name', y='avg_latency',
                    color='model_name',
                    labels={'model_name': 'Model', 'avg_latency': 'Avg Latency (s)'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig2.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Full Rankings Table")
            display = leaderboard.rename(columns={
                'model_name': 'Model', 'avg_score': 'Avg Score',
                'avg_latency': 'Latency (s)', 'total_runs': 'Runs',
                'avg_output_tokens': 'Avg Tokens'
            })
            display.insert(0, 'Rank', range(1, len(display) + 1))
            st.dataframe(display, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────
# PAGE: ELO
# ────────────────────────────────────────────
elif page == "⚡ ELO Ratings":
    st.title("⚡ ELO Ratings")
    st.caption("Chess-style ratings — models gain/lose points in every head-to-head matchup")

    with st.expander("ℹ️ What is ELO?"):
        st.markdown("""
        ELO is the same rating system used in chess, where every player has a score starting at 1000.
        - **Win against a stronger model** → gain more points
        - **Lose to a weaker model** → lose more points
        - **Tie** → small exchange based on expected outcome

        After many matches across diverse prompts, the ELO rankings reflect true model quality.
        **A model with 1100 ELO is significantly stronger than one with 900 ELO.**
        """)

    elo_data = get_elo_ratings()

    if not elo_data:
        st.info("👋 No ELO data yet. Run a benchmark first!")
    else:
        elo_df = pd.DataFrame(elo_data)

        top = elo_df.iloc[0]
        st.success(f"👑 Current champion: **{top['model_name']}** with **{top['elo_score']:.0f} ELO**")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                elo_df, x='model_name', y='elo_score',
                color='model_name', title='ELO Ratings (baseline = 1000)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.add_hline(y=1000, line_dash="dash",
                         line_color="gray", annotation_text="Baseline 1000")
            fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Win / Loss Record")
            record = elo_df[['model_name', 'wins', 'losses', 'ties',
                             'total_matches', 'elo_score']].copy()
            record['win_rate'] = (
                record['wins'] / record['total_matches'].replace(0, 1) * 100
            ).round(1)
            record = record.rename(columns={
                'model_name': 'Model', 'wins': 'W',
                'losses': 'L', 'ties': 'T',
                'total_matches': 'Matches',
                'elo_score': 'ELO',
                'win_rate': 'Win %'
            })
            st.dataframe(record, use_container_width=True, hide_index=True)

        # ELO over time
        history = get_elo_history(limit=100)
        if history and len(history) > 1:
            st.subheader("ELO Over Time")
            st.caption("Watch how model rankings shift as more prompts are tested")
            hist_df = pd.DataFrame(history)
            time_data = []
            for _, row in hist_df.iterrows():
                time_data.append({'time': row['created_at'],
                                  'model': row['model_a'], 'elo': row['new_elo_a']})
                time_data.append({'time': row['created_at'],
                                  'model': row['model_b'], 'elo': row['new_elo_b']})
            time_df = pd.DataFrame(time_data).sort_values('time')
            fig_t = px.line(
                time_df, x='time', y='elo', color='model',
                markers=True, title='ELO trajectory over time'
            )
            fig_t.add_hline(y=1000, line_dash="dash", line_color="gray")
            fig_t.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_t, use_container_width=True)

# ────────────────────────────────────────────
# PAGE: RUN BENCHMARK
# ────────────────────────────────────────────
elif page == "🏁 Run Benchmark":
    st.title("🏁 Run a Benchmark")

    # Clear guide at the top
    with st.expander("📖 How this works — read first!", expanded=True):
        st.markdown("""
        **What happens when you click Run:**
        1. Your prompt is sent to **all 4 AI models simultaneously**
        2. An independent judge AI **scores each response** on accuracy, clarity and completeness
        3. **ELO ratings update** automatically based on who won each head-to-head
        4. Results appear on the Leaderboard and ELO pages

        **Tips for good benchmarks:**
        - Ask questions that require detailed answers (not yes/no)
        - Include multiple parts: "explain X, give an example, and write code"
        - Try different topics: coding, math, ML concepts, system design
        """)

    # Quick prompt suggestions
    st.subheader("💡 Quick prompts to try")
    quick_prompts = {
        "🐍 Python coding": "Implement a binary search tree in Python with insert, search and delete methods. Include time complexity analysis.",
        "🧠 ML concept": "Explain how attention mechanisms work in transformers. Include the math and a Python example.",
        "🏗️ System design": "Design a URL shortener system. Cover the architecture, database schema, and handle 1M requests/day.",
        "📐 Algorithm": "Explain dynamic programming using the coin change problem. Show the recursive and DP solutions in Python.",
        "🗄️ Database": "Compare SQL vs NoSQL databases. When would you use MongoDB over PostgreSQL? Give real examples.",
    }

    cols = st.columns(len(quick_prompts))
    selected_quick = None
    for i, (label, prompt_text) in enumerate(quick_prompts.items()):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"qp_{i}"):
                selected_quick = prompt_text

    st.divider()

    # Prompt input
    default_prompt = selected_quick or st.session_state.get('quick_prompt', '')
    prompt = st.text_area(
        "✏️ Or write your own prompt:",
        value=default_prompt,
        placeholder="Ask anything that requires a detailed, technical answer...",
        height=120
    )

    system_prompt = st.text_input(
        "System prompt (optional):",
        value="You are a helpful AI assistant. Be clear, accurate and provide complete answers with examples."
    )

    col1, col2 = st.columns(2)
    with col1:
        run_judge = st.checkbox("✅ Auto-judge responses", value=True,
                               help="Scores each response on accuracy, clarity, completeness")
    with col2:
        num_judge_runs = st.slider("Judge passes (more = more consistent)",
                                  min_value=1, max_value=5, value=3)

    if st.button("🚀 Run Benchmark Now", type="primary",
                use_container_width=True, disabled=not prompt.strip()):

        if not prompt.strip():
            st.error("Please enter a prompt first!")
        else:
            progress = st.progress(0, text="Starting benchmark...")

            # Step 1
            progress.progress(10, text="🚀 Sending prompt to all 4 models simultaneously...")
            with st.spinner(""):
                result = run_benchmark_sync(prompt, system_prompt)
            progress.progress(40, text="✅ All models responded!")

            st.success(f"Benchmark complete! Run ID: `{result['run_id']}`")

            # Show results side by side
            st.subheader("📝 Model Responses")
            response_cols = st.columns(2)
            for i, (model, data) in enumerate(result['results'].items()):
                with response_cols[i % 2]:
                    with st.expander(f"**{model}** — {data['latency']}s"):
                        if data['output']:
                            st.markdown(data['output'][:800])
                            if len(data['output']) > 800:
                                st.caption("_(truncated for display)_")
                        else:
                            st.error(f"Failed: {data['error']}")

            if run_judge:
                # Step 2
                progress.progress(50, text=f"⚖️ Judge AI scoring responses ({num_judge_runs} passes)...")
                with st.spinner(""):
                    scores = judge_run_stable(result['run_id'], num_runs=num_judge_runs)
                progress.progress(80, text="✅ Scoring complete!")

                if scores:
                    st.subheader("⚖️ Judge Scores")
                    ranked = sorted(scores.items(),
                                  key=lambda x: x[1].get('overall', 0), reverse=True)

                    medals = ["🥇", "🥈", "🥉", "4️⃣"]
                    for i, (model, s) in enumerate(ranked):
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        with col1:
                            st.markdown(f"{medals[i]} **{model}**")
                            st.caption(s.get('reasoning', ''))
                        with col2:
                            st.metric("Overall", f"{s['overall']}/10")
                        with col3:
                            st.metric("Accuracy", f"{s['accuracy']}/10")
                        with col4:
                            st.metric("Clarity", f"{s['clarity']}/10")
                        with col5:
                            st.metric("Complete", f"{s['completeness']}/10")

                    # Radar chart
                    fig_radar = go.Figure()
                    cats = ['Accuracy', 'Clarity', 'Completeness']
                    for model, s in scores.items():
                        vals = [s.get('accuracy', 0),
                               s.get('clarity', 0),
                               s.get('completeness', 0)]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]],
                            theta=cats + [cats[0]],
                            fill='toself', name=model, opacity=0.6
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 10])),
                        title="Score breakdown by dimension"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    # Step 3 — ELO
                    progress.progress(90, text="⚡ Updating ELO ratings...")
                    from app.elo.elo_system import update_elo_for_run
                    elo_update = update_elo_for_run(result['run_id'])
                    progress.progress(100, text="✅ All done!")

                    if elo_update:
                        st.subheader("⚡ Updated ELO Rankings")
                        st.caption("Based on all benchmark runs so far")
                        elo_data = get_elo_ratings()
                        if elo_data:
                            ecols = st.columns(len(elo_data))
                            for i, (col, r) in enumerate(zip(ecols, elo_data)):
                                with col:
                                    st.metric(
                                        r['model_name'],
                                        f"{r['elo_score']:.0f}",
                                        delta=f"{r['wins']}W {r['losses']}L"
                                    )

                    st.balloons()
                    st.success("🎉 Done! Check the **Leaderboard** and **ELO Ratings** pages to see updated rankings.")

    st.divider()

    # Agent section
    st.subheader("🤖 Self-Improving Agent")

    with st.expander("What does the agent do?"):
        st.markdown("""
        The agent is an AI that makes the benchmark **smarter over time**:

        1. **Analyzes** — looks at which model has the lowest ELO and what dimension it struggles with
        2. **Generates** — creates new harder prompts specifically designed to test those weak spots
        3. **Benchmarks** — runs the full pipeline (benchmark + judge + ELO) on each new prompt automatically
        4. **Updates** — ELO ratings reflect the harder, more diverse prompt set

        After running the agent a few times, your ELO rankings become much more trustworthy
        because they're based on many diverse, challenging prompts — not just easy ones.
        """)

    col1, col2 = st.columns(2)
    with col1:
        num_agent_prompts = st.slider(
            "How many new prompts should the agent generate?",
            min_value=1, max_value=5, value=2,
            help="Each prompt runs the full benchmark + judge + ELO pipeline"
        )
    with col2:
        st.info(f"⏱ Estimated time: **{num_agent_prompts * 3}–{num_agent_prompts * 4} minutes**")
        st.caption("The agent runs everything automatically — just wait for it to finish")

    if st.button("🤖 Run Self-Improving Agent", use_container_width=True):
        if df.empty:
            st.error("❌ Run at least one manual benchmark first so the agent has data to analyze!")
        else:
            from app.agent.improvement_agent import run_improvement_cycle

            with st.spinner(f"Agent thinking and running {num_agent_prompts} benchmark cycles... this takes a few minutes"):
                cycle_results = asyncio.run(
                    run_improvement_cycle(num_prompts=num_agent_prompts)
                )

            if cycle_results:
                st.success(f"✅ Agent complete! Ran {len(cycle_results['prompts_tested'])} new benchmarks")

                st.subheader("Prompts the agent generated:")
                for i, p in enumerate(cycle_results['prompts_tested'], 1):
                    st.markdown(f"**{i}.** {p[:120]}...")

                st.subheader("Updated ELO after agent run:")
                final = cycle_results.get('final_ratings', [])
                medals = ["🥇", "🥈", "🥉", "4️⃣"]
                for i, r in enumerate(final):
                    st.markdown(
                        f"{medals[i]} **{r['model_name']}**: "
                        f"`{r['elo_score']:.0f} ELO` — "
                        f"{r['wins']}W / {r['losses']}L / {r['ties']}T"
                    )
            else:
                st.error("Agent failed — make sure you have existing benchmark data first")

# ────────────────────────────────────────────
# PAGE: ANALYTICS
# ────────────────────────────────────────────
elif page == "📊 Analytics":
    st.title("📊 Analytics")
    st.caption("Deep dive into model performance patterns")

    if df.empty:
        st.info("👋 No data yet. Run some benchmarks first!")
    else:
        scored_df = df.dropna(subset=['judge_score'])

        if scored_df.empty:
            st.info("No judged results yet.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Score Distribution")
                st.caption("Tighter box = more consistent model")
                fig_box = px.box(
                    scored_df, x='model_name', y='judge_score',
                    color='model_name',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_box.update_layout(
                    showlegend=False, yaxis_range=[0, 10],
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col2:
                st.subheader("Speed vs Quality")
                st.caption("Top-left = fast AND good. Ideal position.")
                fig_sc = px.scatter(
                    scored_df, x='latency_seconds', y='judge_score',
                    color='model_name', size='output_tokens',
                    labels={
                        'latency_seconds': 'Latency (s)',
                        'judge_score': 'Score',
                        'model_name': 'Model'
                    },
                    hover_data=['prompt']
                )
                fig_sc.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_sc, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Token Usage")
                st.caption("How verbose is each model?")
                tok_df = df.groupby('model_name')['output_tokens'].mean().reset_index()
                fig_tok = px.bar(
                    tok_df, x='model_name', y='output_tokens',
                    color='model_name',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_tok.update_layout(showlegend=False)
                st.plotly_chart(fig_tok, use_container_width=True)

            with col2:
                st.subheader("Score Over Time")
                st.caption("Are models improving or degrading?")
                fig_time = px.line(
                    scored_df.sort_values('created_at'),
                    x='created_at', y='judge_score',
                    color='model_name',
                    markers=True
                )
                fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_time, use_container_width=True)

# ────────────────────────────────────────────
# PAGE: HISTORY
# ────────────────────────────────────────────
elif page == "📋 History":
    st.title("📋 Benchmark History")
    st.caption("Every prompt ever tested on this platform")

    if df.empty:
        st.info("👋 No history yet. Run your first benchmark!")
    else:
        runs = df[['run_id', 'prompt', 'created_at']].drop_duplicates('run_id')
        runs = runs.sort_values('created_at', ascending=False)

        st.metric("Total benchmark runs", len(runs))
        st.divider()

        for _, run in runs.iterrows():
            with st.expander(f"🏁 `{run['run_id']}` — {run['prompt'][:70]}..."):
                st.caption(f"Run at: {run['created_at']}")
                run_results = get_results_by_run(run['run_id'])

                if run_results:
                    run_df = pd.DataFrame(run_results)
                    scored = run_df.dropna(subset=['judge_score'])

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        for r in run_results:
                            score_str = (f"Score: **{r['judge_score']}/10**"
                                        if r['judge_score'] else "Not judged")
                            st.markdown(
                                f"**{r['model_name']}** — "
                                f"{r['latency_seconds']}s — {score_str}"
                            )
                            if r.get('judge_reasoning'):
                                st.caption(f"_{r['judge_reasoning']}_")
                    with col2:
                        if not scored.empty:
                            fig = px.bar(
                                scored, x='model_name', y='judge_score',
                                color='model_name',
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig.update_layout(
                                showlegend=False,
                                yaxis_range=[0, 10],
                                height=250,
                                margin=dict(t=10, b=10)
                            )
                            st.plotly_chart(fig, use_container_width=True)