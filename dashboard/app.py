# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PIL import Image

# =====================
# Custom CSS for Engineering Dashboard Theme (Responsive)
# =====================
custom_css = """
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e8e8e8;
}
.metric-card {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    border: 1px solid #4a5568;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    padding: 1.5rem 1rem 1rem 1rem;
    margin-bottom: 1rem;
    text-align: center;
    min-width: 0;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
}
.metric-label {
    color: #a0aec0;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    color: #00d4ff;
    font-size: 2.2rem;
    font-weight: 700;
    word-break: break-word;
    font-family: 'Courier New', monospace;
}
.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    color: #00d4ff;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #00d4ff;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-transform: uppercase;
    border-bottom: 2px solid #00d4ff;
    padding-bottom: 0.5rem;
}
hr {
    border: none;
    border-top: 2px solid #4a5568;
    margin: 2rem 0 1.5rem 0;
}
.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
}
.status-online { background-color: #48bb78; }
.status-warning { background-color: #ed8936; }
.status-offline { background-color: #f56565; }
.engineering-header {
    background: linear-gradient(90deg, #1a202c 0%, #2d3748 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #4a5568;
    margin-bottom: 2rem;
}
.system-status {
    background: #2d3748;
    border: 1px solid #4a5568;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    width: 100%;
    box-sizing: border-box;
}
/* Responsive Design */
@media (max-width: 1200px) {
    .section-header { font-size: 1.2rem; }
    .metric-value { font-size: 2rem; }
    .metric-label { font-size: 0.85rem; }
    .metric-icon { font-size: 1.6rem; }
}
@media (max-width: 900px) {
    .section-header { font-size: 1.1rem; }
    .metric-value { font-size: 1.8rem; }
    .metric-label { font-size: 0.8rem; }
    .metric-icon { font-size: 1.4rem; }
    .metric-card { padding: 1.2rem 0.8rem 0.8rem 0.8rem; }
    .engineering-header { padding: 0.8rem; }
    .system-status { padding: 0.8rem; }
}
@media (max-width: 768px) {
    .section-header { font-size: 1rem; }
    .metric-value { font-size: 1.5rem; }
    .metric-label { font-size: 0.75rem; }
    .metric-icon { font-size: 1.2rem; }
    .metric-card { padding: 1rem 0.5rem 0.5rem 0.5rem; }
    .engineering-header { padding: 0.6rem; }
    .system-status { padding: 0.6rem; }
    .engineering-header > div > div:first-child { font-size: 1.8rem !important; }
    .engineering-header > div > div:last-child { font-size: 0.9rem !important; }
}
@media (max-width: 480px) {
    .section-header { font-size: 0.9rem; }
    .metric-value { font-size: 1.3rem; }
    .metric-label { font-size: 0.7rem; }
    .metric-icon { font-size: 1rem; }
    .metric-card { padding: 0.8rem 0.4rem 0.4rem 0.4rem; }
    .engineering-header { padding: 0.5rem; }
    .system-status { padding: 0.5rem; }
    .engineering-header > div > div:first-child { font-size: 1.5rem !important; }
    .engineering-header > div > div:last-child { font-size: 0.8rem !important; }
}
/* Ensure all charts and tables are responsive */
.stPlotlyChart, .stDataFrame {
    width: 100% !important;
    max-width: 100% !important;
}
/* Make sidebar responsive */
.sidebar .sidebar-content {
    width: 100% !important;
}
@media (max-width: 768px) {
    .sidebar .sidebar-content {
        padding: 0.5rem !important;
    }
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =====================
# Sidebar
# =====================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/Bitmap_ERCOT_logo.png", use_container_width=True)
st.sidebar.markdown("""
### System Control Panel
**Wind Energy Management System**

- **Project:** ERCOT Wind Demo
- **Phase:** Production
- **Version:** 2.1.0

*Engineering Dashboard v2.1*
""")

# =====================
# Set paths
# =====================
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def load_csv(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        return None

def load_image(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    try:
        return Image.open(path)
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        return None

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="Engineering Dashboard - Wind Energy Management",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# Header
# =====================
st.markdown("""
<div class='engineering-header'>
    <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;'>
        <div style='font-size:2.2rem; font-weight:800; color:#00d4ff; display: flex; align-items: center; gap: 0.7rem; font-family: "Courier New", monospace; flex: 1; min-width: 0;'>
            <span style='font-size:2.5rem;'>⚡</span> ENGINEERING DASHBOARD - WIND ENERGY MANAGEMENT SYSTEM
        </div>
        <div style='font-size:1.1rem; color:#a0aec0; font-family: "Courier New", monospace; white-space: nowrap;'>
            SYSTEM STATUS: ONLINE | Last updated: {}
        </div>
    </div>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# =====================
# System Status Panel (Responsive)
# =====================
# Use responsive columns that stack on mobile
status_cols = st.columns([1, 1, 1, 1])
with status_cols[0]:
    st.markdown("""
    <div class='system-status'>
        <div style='display: flex; align-items: center;'>
            <span class='status-indicator status-online'></span>
            <strong>WIND TURBINES</strong>
        </div>
        <div style='color: #48bb78; font-size: 1.2rem;'>OPERATIONAL</div>
    </div>
    """, unsafe_allow_html=True)
with status_cols[1]:
    st.markdown("""
    <div class='system-status'>
        <div style='display: flex; align-items: center;'>
            <span class='status-indicator status-online'></span>
            <strong>BATTERY STORAGE</strong>
        </div>
        <div style='color: #48bb78; font-size: 1.2rem;'>ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)
with status_cols[2]:
    st.markdown("""
    <div class='system-status'>
        <div style='display: flex; align-items: center;'>
            <span class='status-indicator status-online'></span>
            <strong>GRID CONNECTION</strong>
        </div>
        <div style='color: #48bb78; font-size: 1.2rem;'>STABLE</div>
    </div>
    """, unsafe_allow_html=True)
with status_cols[3]:
    st.markdown("""
    <div class='system-status'>
        <div style='display: flex; align-items: center;'>
            <span class='status-indicator status-warning'></span>
            <strong>AI AGENTS</strong>
        </div>
        <div style='color: #ed8936; font-size: 1.2rem;'>LEARNING</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =====================
# Load Data
# =====================
eval_metrics = load_csv('evaluation_metrics.csv')
emissions = load_csv('emissions_reduction.csv')
cost_savings = load_csv('cost_savings.csv')
battery_soc = load_csv('battery_soc.csv')
load_vs_gen = load_csv('load_vs_generation.csv')
rl_convergence = load_csv('rl_convergence.csv')
battery_cycles = load_csv('battery_cycles.csv')
agent_action_dist = load_csv('agent_action_distribution.csv')
training_plot = load_image('training_plot.png')

# =====================
# Key Performance Indicators (Responsive)
# =====================
st.markdown("<div class='section-header'>📊 KEY PERFORMANCE INDICATORS</div>", unsafe_allow_html=True)
metric_icons = [
    "⚡", "🔄", "🔋", "📉", "🌱", "🏔️", "⏱️"
]
metric_labels = [
    "AVERAGE REWARD",
    "GRID BALANCE ERROR (MWH)",
    "STORAGE UTILIZATION (%)",
    "CURTAILMENT REDUCTION (%)",
    "EMISSIONS REDUCTION (TONS CO₂)",
    "PEAK LOAD COVERAGE (%)",
    "SYSTEM UPTIME (%)"
]
metric_values = [
    eval_metrics['average_reward'].iloc[0] if eval_metrics is not None and 'average_reward' in eval_metrics.columns else "N/A",
    eval_metrics['grid_balance_error'].iloc[0] if eval_metrics is not None and 'grid_balance_error' in eval_metrics.columns else "N/A",
    eval_metrics['storage_utilization'].iloc[0] if eval_metrics is not None and 'storage_utilization' in eval_metrics.columns else "N/A",
    eval_metrics['curtailment_reduction'].iloc[0] if eval_metrics is not None and 'curtailment_reduction' in eval_metrics.columns else "N/A",
    emissions['emissions_reduction'].iloc[0] if emissions is not None and 'emissions_reduction' in emissions.columns else "N/A",
    eval_metrics['peak_load_coverage'].iloc[0] if eval_metrics is not None and 'peak_load_coverage' in eval_metrics.columns else "N/A",
    eval_metrics['system_uptime'].iloc[0] if eval_metrics is not None and 'system_uptime' in eval_metrics.columns else "N/A"
]

# Format average reward to 1 decimal place if possible
avg_reward = metric_values[0]
try:
    avg_reward = float(avg_reward)
    avg_reward = f"{avg_reward:.1f}"
except Exception:
    pass
metric_values[0] = avg_reward

# Responsive metric cards - use container for better mobile layout
with st.container():
    # First row of metrics
    metric_cols = st.columns(4)
    for i in range(4):
        with metric_cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-icon'>{metric_icons[i]}</div>
                <div class='metric-label'>{metric_labels[i]}</div>
                <div class='metric-value'>{metric_values[i]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row of metrics
    metric_cols2 = st.columns(4)
    for i in range(4, 7):
        with metric_cols2[i-4]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-icon'>{metric_icons[i]}</div>
                <div class='metric-label'>{metric_labels[i]}</div>
                <div class='metric-value'>{metric_values[i]}</div>
            </div>
            """, unsafe_allow_html=True)
        if i == 6:
            st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =====================
# System Monitoring (Responsive)
# =====================
with st.container():
    st.markdown("<div class='section-header'>🔍 SYSTEM MONITORING</div>", unsafe_allow_html=True)
    if training_plot:
        st.image(training_plot, caption="TRAINING PROGRESS ANALYSIS", use_container_width=True)
    else:
        st.info("Training plot not available.")

    st.markdown("<div class='section-header'>🔋 BATTERY STATE OF CHARGE (SOC) TREND</div>", unsafe_allow_html=True)
    if battery_soc is not None and 'timestamp' in battery_soc.columns and 'soc' in battery_soc.columns:
        st.line_chart(battery_soc.set_index('timestamp')['soc'], use_container_width=True)
    else:
        st.info("Battery SOC data not available.")

    st.markdown("<div class='section-header'>⚡ LOAD DEMAND VS GENERATION PROFILES</div>", unsafe_allow_html=True)
    if load_vs_gen is not None and {'timestamp', 'load', 'generation'}.issubset(load_vs_gen.columns):
        st.line_chart(load_vs_gen.set_index('timestamp')[['load', 'generation']], use_container_width=True)
    else:
        st.info("Load vs Generation data not available.")

    st.markdown("<div class='section-header'>🧠 RL CONVERGENCE (AVERAGE REWARD PER EPISODE)</div>", unsafe_allow_html=True)
    if rl_convergence is not None and {'episode', 'average_reward'}.issubset(rl_convergence.columns):
        st.line_chart(rl_convergence.set_index('episode')['average_reward'], use_container_width=True)
    else:
        st.info("RL convergence data not available.")

st.markdown("<hr>", unsafe_allow_html=True)

# =====================
# Data Analysis (Responsive)
# =====================
with st.container():
    st.markdown("<div class='section-header'>📈 DATA ANALYSIS</div>", unsafe_allow_html=True)
    if cost_savings is not None:
        st.dataframe(cost_savings, use_container_width=True)
    else:
        st.info("Cost savings data not available.")

    st.markdown("<div class='section-header'>🔄 BATTERY CYCLING COUNTS</div>", unsafe_allow_html=True)
    if battery_cycles is not None:
        st.dataframe(battery_cycles, use_container_width=True)
    else:
        st.info("Battery cycling data not available.")

st.markdown("<hr>", unsafe_allow_html=True)

# =====================
# Agent Behavior Analysis (Responsive)
# =====================
with st.container():
    st.markdown("<div class='section-header'>🤖 AGENT BEHAVIOR ANALYSIS</div>", unsafe_allow_html=True)
    if agent_action_dist is not None and 'action' in agent_action_dist.columns and 'count' in agent_action_dist.columns:
        st.bar_chart(agent_action_dist.set_index('action')['count'], use_container_width=True)
    else:
        st.info("Agent action distribution data not available.")

st.markdown("<hr>", unsafe_allow_html=True)

# =====================
# Engineering Notes (Responsive)
# =====================
with st.container():
    st.markdown("<div class='section-header'>📝 ENGINEERING NOTES & SYSTEM LOGS</div>", unsafe_allow_html=True)
    notes_path = os.path.join(OUTPUTS_DIR, "project_notes.txt")
    if os.path.exists(notes_path):
        with open(notes_path, "r") as f:
            notes = f.read()
        st.text_area("SYSTEM LOGS / ENGINEERING NOTES", notes, height=150)
    else:
        st.text_area("SYSTEM LOGS / ENGINEERING NOTES", "No system logs available.", height=150)

st.markdown("<hr>", unsafe_allow_html=True)

# =====================
# Footer (Responsive)
# =====================
st.markdown("""
<div style='text-align:center; color:#a0aec0; font-size:1rem; margin-top:2rem; font-family: "Courier New", monospace; word-wrap: break-word;'>
    ENGINEERING DASHBOARD v2.1 | ERCOT WIND & BMS PROJECT TEAM | © 2024
</div>
""", unsafe_allow_html=True) 