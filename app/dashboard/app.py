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
from app.database.db import init_db, get_all_results, get_results_by_run, get_elo_ratings, get_elo_history, init_elo_table
from app.runner.prompt_runner import run_benchmark
from app.evaluator.judge import judge_run_stable

# ─────────────────────────────────────────────
# PAGE CONFIG
# Must be the FIRST streamlit call in the file
# Sets browser tab title, icon, and layout
# layout="wide" = uses full browser width (better for dashboards)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoEval Arena",
    page_icon="🏆",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# Streamlit has default styling but we override some parts
# to make it look more professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .rank-1 { color: #FFD700; font-size: 1.5rem; }
    .rank-2 { color: #C0C0C0; font-size: 1.5rem; }
    .rank-3 { color: #CD7F32; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_data():
    """
    Loads all benchmark results from SQLite into a pandas DataFrame.
    
    DataFrame = like an Excel spreadsheet in Python
    Every row = one model's response to one prompt
    We use it because plotly charts expect DataFrames as input
    """
    init_db()
    results = get_all_results()
    if not results:
        return pd.DataFrame()
    # pd.DataFrame(list_of_dicts) converts our SQLite rows
    # directly into a DataFrame — each dict key becomes a column
    return pd.DataFrame(results)

def run_benchmark_sync(prompt: str, system_prompt: str = None):
    """
    Streamlit runs synchronously but our benchmark is async.
    This wrapper bridges the gap.
    
    asyncio.run() creates a new event loop, runs the async function,
    returns the result, then closes the loop.
    We can't use "await" directly in Streamlit — this is the workaround.
    """
    return asyncio.run(run_benchmark(prompt, system_prompt))

def get_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates average scores per model across ALL benchmark runs.
    
    groupby('model_name') = group rows by model
    ['judge_score'].mean() = average the judge_score column per group
    reset_index() = turns the grouped result back into a normal DataFrame
    sort_values() = sort by score descending
    dropna() = remove models that have no judge scores yet
    """
    if df.empty or 'judge_score' not in df.columns:
        return pd.DataFrame()
    
    leaderboard = df.dropna(subset=['judge_score']).groupby('model_name').agg(
        avg_score=('judge_score', 'mean'),       # average judge score
        avg_latency=('latency_seconds', 'mean'), # average response time
        total_runs=('run_id', 'nunique'),        # how many unique runs
        avg_input_tokens=('input_tokens', 'mean'),
        avg_output_tokens=('output_tokens', 'mean')
    ).reset_index()
    
    # Round for display
    leaderboard['avg_score'] = leaderboard['avg_score'].round(2)
    leaderboard['avg_latency'] = leaderboard['avg_latency'].round(3)
    
    return leaderboard.sort_values('avg_score', ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# SIDEBAR
# Everything in "with st.sidebar:" appears in the left panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/twitter/376/trophy_1f3c6.png", width=60)
    st.title("AutoEval Arena")
    st.caption("LLM Benchmarking Platform")
    st.divider()
    
    # Navigation — st.radio creates a radio button group
    # The selected option controls what the main panel shows
    page = st.radio(
    "Navigate",
    ["🏆 Leaderboard", "⚡ ELO Ratings", "🏁 Run Benchmark", "📊 Analytics", "📋 History"],
    label_visibility="collapsed"
    )   
    
    st.divider()
    st.caption("Built with Groq + Streamlit")
    st.caption("Judge: GPT-OSS 120B")

# Load all data once — used across all pages
df = load_data()

# ─────────────────────────────────────────────
# PAGE 1 — LEADERBOARD
# ─────────────────────────────────────────────
if page == "🏆 Leaderboard":
    st.markdown('<p class="main-header">🏆 AutoEval Arena Leaderboard</p>', unsafe_allow_html=True)
    
    if df.empty:
        # st.info shows a blue info box
        st.info("No benchmark results yet. Go to 'Run Benchmark' to get started!")
    else:
        leaderboard = get_leaderboard(df)
        
        if leaderboard.empty:
            st.info("No judged results yet. Run the judge after benchmarking.")
        else:
            # ── Top 3 podium ──
            # st.columns splits the page into equal columns side by side
            col1, col2, col3 = st.columns(3)
            
            medals = ["🥇", "🥈", "🥉"]
            colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
            
            for i, col in enumerate([col1, col2, col3]):
                if i < len(leaderboard):
                    row = leaderboard.iloc[i]
                    # iloc[i] = get row by position (0=first, 1=second etc)
                    with col:
                        st.markdown(f"""
                        <div style='text-align:center; padding:1rem; 
                             border:2px solid {colors[i]}; border-radius:12px;'>
                            <div style='font-size:2rem'>{medals[i]}</div>
                            <div style='font-size:1.1rem; font-weight:bold'>
                                {row['model_name']}
                            </div>
                            <div style='font-size:2rem; color:{colors[i]}; font-weight:bold'>
                                {row['avg_score']}/10
                            </div>
                            <div style='color:gray; font-size:0.85rem'>
                                ⚡ {row['avg_latency']}s avg latency
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
            
            # ── Full leaderboard table ──
            st.subheader("Full Rankings")
            
            # Rename columns for display
            display_df = leaderboard.rename(columns={
                'model_name': 'Model',
                'avg_score': 'Avg Score',
                'avg_latency': 'Avg Latency (s)',
                'total_runs': 'Total Runs',
                'avg_input_tokens': 'Avg Input Tokens',
                'avg_output_tokens': 'Avg Output Tokens'
            })
            
            # Add rank column
            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
            
            # st.dataframe shows an interactive sortable table
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # ── Score bar chart ──
            st.subheader("Score Comparison")
            
            # px.bar = plotly express bar chart
            # x = which column goes on x axis
            # y = which column goes on y axis
            # color = which column determines bar color
            fig_scores = px.bar(
                leaderboard,
                x='model_name',
                y='avg_score',
                color='model_name',
                title='Average Judge Score by Model',
                labels={'model_name': 'Model', 'avg_score': 'Average Score'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_scores.update_layout(
                showlegend=False,
                yaxis_range=[0, 10],  # always show full 0-10 scale
                plot_bgcolor='rgba(0,0,0,0)',  # transparent background
            )
            # st.plotly_chart renders an interactive plotly chart
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # ── Latency bar chart ──
            st.subheader("Latency Comparison")
            fig_latency = px.bar(
                leaderboard,
                x='model_name',
                y='avg_latency',
                color='model_name',
                title='Average Response Latency by Model (lower is better)',
                labels={'model_name': 'Model', 'avg_latency': 'Avg Latency (s)'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_latency.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_latency, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2 — RUN BENCHMARK
# ─────────────────────────────────────────────
elif page == "🏁 Run Benchmark":
    st.title("🏁 Run New Benchmark")
    st.caption("Send a prompt to all models simultaneously and judge the results")
    
    # st.text_area = multiline text input box
    # height=100 = how tall the box appears
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="e.g. Explain how neural networks learn...",
        height=100
    )
    
    system_prompt = st.text_input(
        "System prompt (optional):",
        value="You are a helpful AI assistant. Be clear and concise."
    )
    
    # Two columns for the options
    col1, col2 = st.columns(2)
    with col1:
        # st.checkbox = a toggle checkbox
        run_judge = st.checkbox("Run judge after benchmark", value=True)
    with col2:
        num_judge_runs = st.slider(
            "Judge ensemble runs",
            min_value=1, max_value=5, value=3
        )
        # st.slider = a draggable slider for numbers
    
    # st.button = clickable button, returns True when clicked
    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        if not prompt.strip():
            # st.error = red error box
            st.error("Please enter a prompt first!")
        else:
            # st.spinner = shows a loading spinner while code runs
            with st.spinner("Running benchmark across all models..."):
                result = run_benchmark_sync(prompt, system_prompt)
            
            st.success(f"✅ Benchmark complete! Run ID: {result['run_id']}")
            
            # Show raw results in an expander
            # st.expander = collapsible section
            with st.expander("View raw results"):
                for model, data in result['results'].items():
                    st.markdown(f"**{model}** — {data['latency']}s")
                    st.text(data['output'][:500] if data['output'] else "Failed")
                    st.divider()
            
            if run_judge:
                with st.spinner(f"Running judge ({num_judge_runs} passes)..."):
                    scores = judge_run_stable(result['run_id'], num_runs=num_judge_runs)
                
                if scores:
                    st.subheader("Judge Scores")
                    # Sort by overall score
                    ranked = sorted(
                        scores.items(),
                        key=lambda x: x[1].get('overall', 0),
                        reverse=True
                    )
                    for rank, (model, s) in enumerate(ranked, 1):
                        # st.metric = a big number display card
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"#{rank} {model}", f"{s['overall']}/10")
                        with col2:
                            st.metric("Accuracy", f"{s['accuracy']}/10")
                        with col3:
                            st.metric("Clarity", f"{s['clarity']}/10")
                        with col4:
                            st.metric("Completeness", f"{s['completeness']}/10")
                        st.caption(f"Reasoning: {s.get('reasoning', '')}")
                        st.divider()
                    
                    # Radar chart — shows all 3 dimensions per model
                    st.subheader("Score Radar Chart")
                    
                    fig_radar = go.Figure()
                    # Categories = the 3 dimensions we score on
                    categories = ['Accuracy', 'Clarity', 'Completeness']
                    
                    for model, s in scores.items():
                        values = [
                            s.get('accuracy', 0),
                            s.get('clarity', 0),
                            s.get('completeness', 0)
                        ]
                        # go.Scatterpolar = radar/spider chart in plotly
                        # r = the values (how far from center)
                        # theta = the labels around the radar
                        # fill='toself' = fills the polygon
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values + [values[0]],  # repeat first value to close polygon
                            theta=categories + [categories[0]],
                            fill='toself',
                            name=model,
                            opacity=0.6
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 10])),
                        showlegend=True,
                        title="Score Dimensions per Model"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 — ANALYTICS
# ─────────────────────────────────────────────
elif page == "📊 Analytics":
    st.title("📊 Analytics")
    
    if df.empty:
        st.info("No data yet. Run some benchmarks first!")
    else:
        # ── Score distribution ──
        st.subheader("Score Distribution by Model")
        
        scored_df = df.dropna(subset=['judge_score'])
        
        if not scored_df.empty:
            # px.box = box plot showing score distribution
            # Box plot shows: min, Q1, median, Q3, max per model
            # Great for seeing consistency vs variance
            fig_box = px.box(
                scored_df,
                x='model_name',
                y='judge_score',
                color='model_name',
                title='Score Distribution (consistency = tight box)',
                labels={'model_name': 'Model', 'judge_score': 'Judge Score'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_box.update_layout(
                showlegend=False,
                yaxis_range=[0, 10],
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("Tighter box = more consistent model. Wider box = more variable.")
        
        # ── Latency vs Score scatter ──
        st.subheader("Latency vs Score (the sweet spot)")
        
        if not scored_df.empty:
            # px.scatter = scatter plot
            # Each dot = one benchmark result
            # x = latency, y = score
            # color = which model
            # size = output tokens (bigger dot = longer response)
            fig_scatter = px.scatter(
                scored_df,
                x='latency_seconds',
                y='judge_score',
                color='model_name',
                size='output_tokens',
                title='Speed vs Quality (top-left = best)',
                labels={
                    'latency_seconds': 'Latency (s)',
                    'judge_score': 'Judge Score',
                    'model_name': 'Model'
                },
                hover_data=['prompt']
            )
            fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Top-left corner = fast AND high quality. Ideal model position.")
        
        # ── Token usage ──
        st.subheader("Token Usage by Model")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_tokens = px.bar(
                df.groupby('model_name')['output_tokens'].mean().reset_index(),
                x='model_name',
                y='output_tokens',
                title='Avg Output Tokens (verbosity)',
                color='model_name',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_tokens.update_layout(showlegend=False)
            st.plotly_chart(fig_tokens, use_container_width=True)
        
        with col2:
            # Score over time — shows if models improve/degrade
            if 'created_at' in df.columns and not scored_df.empty:
                fig_time = px.line(
                    scored_df.sort_values('created_at'),
                    x='created_at',
                    y='judge_score',
                    color='model_name',
                    title='Score Over Time',
                    labels={'created_at': 'Time', 'judge_score': 'Score'}
                )
                st.plotly_chart(fig_time, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — HISTORY
# ─────────────────────────────────────────────
elif page == "📋 History":
    st.title("📋 Benchmark History")
    
    if df.empty:
        st.info("No benchmark history yet.")
    else:
        # Get unique runs
        # drop_duplicates = keep only one row per unique run_id
        runs = df[['run_id', 'prompt', 'created_at']].drop_duplicates('run_id')
        runs = runs.sort_values('created_at', ascending=False)
        
        st.caption(f"Total runs: {len(runs)}")
        
        # Show each run as an expander
        for _, run in runs.iterrows():
            # iterrows() = loop through DataFrame rows
            # _ = row index (we don't need it)
            # run = the row as a Series (like a dict)
            
            with st.expander(f"Run {run['run_id']} — {run['prompt'][:60]}..."):
                run_results = get_results_by_run(run['run_id'])
                
                if run_results:
                    run_df = pd.DataFrame(run_results)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Model Responses**")
                        for r in run_results:
                            score_str = f"Score: {r['judge_score']}/10" if r['judge_score'] else "Not judged"
                            st.markdown(f"**{r['model_name']}** — {r['latency_seconds']}s — {score_str}")
                            st.text(str(r['output'])[:300] + "...")
                            st.divider()
                    
                    with col2:
                        # Mini score chart for this run
                        scored = run_df.dropna(subset=['judge_score'])
                        if not scored.empty:
                            fig = px.bar(
                                scored,
                                x='model_name',
                                y='judge_score',
                                title='Scores this run',
                                color='model_name',
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig.update_layout(
                                showlegend=False,
                                yaxis_range=[0, 10],
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                      

# ─────────────────────────────────────────────
# PAGE 5 — ELO RATINGS
# ─────────────────────────────────────────────  

elif page == "⚡ ELO Ratings":
    st.title("⚡ ELO Ratings")
    st.caption("Chess-style rating system — updated after every benchmark run")

    init_elo_table()
    elo_data = get_elo_ratings()

    if not elo_data:
        st.info("No ELO data yet. Run a benchmark with judging to generate ELO ratings.")
    else:
        elo_df = pd.DataFrame(elo_data)

        # ── Top model highlight ──
        top = elo_df.iloc[0]
        st.success(f"👑 Current #1: **{top['model_name']}** with **{top['elo_score']:.1f} ELO**")

        st.divider()

        # ── ELO scores bar chart ──
        st.subheader("ELO Ratings")
        fig_elo = px.bar(
            elo_df,
            x='model_name',
            y='elo_score',
            color='model_name',
            title='Current ELO Ratings (baseline = 1000)',
            labels={'model_name': 'Model', 'elo_score': 'ELO Score'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        # Add baseline reference line at 1000
        fig_elo.add_hline(
            y=1000,
            line_dash="dash",
            line_color="gray",
            annotation_text="Baseline (1000)"
        )
        fig_elo.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_elo, use_container_width=True)

        # ── Win/Loss/Tie record ──
        st.subheader("Win / Loss / Tie Record")
        record_df = elo_df[['model_name', 'wins', 'losses', 'ties', 'total_matches', 'elo_score']].copy()
        record_df['win_rate'] = (record_df['wins'] / record_df['total_matches'] * 100).round(1)
        record_df = record_df.rename(columns={
            'model_name': 'Model',
            'wins': 'Wins',
            'losses': 'Losses',
            'ties': 'Ties',
            'total_matches': 'Total',
            'elo_score': 'ELO',
            'win_rate': 'Win Rate %'
        })
        st.dataframe(record_df, use_container_width=True, hide_index=True)

        # ── ELO history ──
        st.subheader("Recent Match History")
        history = get_elo_history(limit=20)
        if history:
            history_df = pd.DataFrame(history)
            history_df['result'] = history_df['winner'].apply(
                lambda w: 'Tie' if w is None else f"{w} won"
            )
            display_history = history_df[[
                'model_a', 'model_b', 'result',
                'elo_change_a', 'elo_change_b',
                'new_elo_a', 'new_elo_b', 'created_at'
            ]].rename(columns={
                'model_a': 'Model A',
                'model_b': 'Model B',
                'result': 'Result',
                'elo_change_a': 'Change A',
                'elo_change_b': 'Change B',
                'new_elo_a': 'New ELO A',
                'new_elo_b': 'New ELO B',
                'created_at': 'Time'
            })
            st.dataframe(display_history, use_container_width=True, hide_index=True)

        # # ── ELO over time line chart ──
        # if history:
        #     st.subheader("ELO Over Time")
        #     history_df['created_at'] = pd.to_datetime(history_df['created_at'])

        #     # Build a time series for each model
        #     # We need to melt the history into model/elo pairs per timestamp
        #     time_data = []
        #     for _, row in history_df.iterrows():
        #         time_data.append({
        #             'time': row['created_at'],
        #             'model': row['model_a'],
        #             'elo': row['new_elo_a']
        #         })
        #         time_data.append({
        #             'time': row['created_at'],
        #             'model': row['model_b'],
        #             'elo': row['new_elo_b']
        #         })

        #     time_df = pd.DataFrame(time_data).sort_values('time')
        #     fig_time = px.line(
        #         time_df,
        #         x='time',
        #         y='elo',
        #         color='model',
        #         title='ELO Rating Over Time',
        #         markers=True,
        #         labels={'time': 'Time', 'elo': 'ELO Score', 'model': 'Model'}
        #     )
        #     fig_time.add_hline(y=1000, line_dash="dash", line_color="gray")
        #     fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        #     st.plotly_chart(fig_time, use_container_width=True)