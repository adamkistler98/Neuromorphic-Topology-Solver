import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time

# --- 1. THE NUCLEAR OPTION: GLOBAL DARK PLOTTING ---
plt.style.use('dark_background') 

# --- 2. STEALTH CONFIGURATION ---
st.set_page_config(
    page_title="Neuromorphic Architect v15", 
    layout="wide", 
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# --- 3. DEEP BLACK CSS INJECTION ---
st.markdown("""
<style>
    /* GLOBAL BLACKOUT */
    .stApp { background-color: #000000; color: #00FF41; }
    
    /* SIDEBAR & INPUTS */
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1A1A1A; }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"], input.st-ai, textarea.st-ai, .stSelectbox div {
        background-color: #080808 !important; border: 1px solid #333 !important; color: #00FF41 !important;
    }
    ul[data-baseweb="menu"], div[data-baseweb="popover"] { background-color: #080808 !important; border: 1px solid #333 !important; }
    li[data-baseweb="option"] { color: #00FF41 !important; }
    
    /* TEXT & METRICS */
    h1, h2, h3, h4 { color: #00FF41 !important; font-family: 'Courier New', monospace; letter-spacing: -1px; margin: 0px; }
    p, label, .stCaption { color: #888 !important; font-family: 'Consolas', monospace; }
    
    /* COMPACT METRICS BOXES */
    div[data-testid="stMetric"] { background-color: #0A0A0A; border: 1px solid #222; padding: 5px !important; border-left: 3px solid #00FF41; }
    div[data-testid="stMetricValue"] { font-size: 18px !important; color: #00FF41 !important; }
    div[data-testid="stMetricLabel"] { font-size: 10px !important; color: #666 !important; }
    
    /* REMOVE ALL PLOT PADDING/MARGINS */
    .main .block-container { padding: 1rem; }
    div[data-testid="stImage"] { background: transparent !important; margin: 0px; }
    button[title="View fullscreen"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- 4. BIO-ENGINE (UPDATED PHYSICS KERNEL) ---
class BioEngine:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.agents = np.zeros((num_agents, 3))
        self.agents[:, 0] = np.random.uniform(width*0.4, width*0.6, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.4, height*0.6, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, nodes, speed, decay, sensor_angle, noise_level):
        self.steps += 1
        sensor_dist = 9.0
        
        # SENSING
        angles = self.agents[:, 2]
        def get_pos(a):
            x = (self.agents[:, 0] + np.cos(a) * sensor_dist) % self.width
            y = (self.agents[:, 1] + np.sin(a) * sensor_dist) % self.height
            return x.astype(int), y.astype(int)

        lx, ly = get_pos(angles - sensor_angle)
        cx, cy = get_pos(angles)
        rx, ry = get_pos(angles + sensor_angle)
        
        l_val = self.trail_map[ly, lx]
        c_val = self.trail_map[cy, cx]
        r_val = self.trail_map[ry, rx]
        
        # DECISION (With Noise Injection)
        jitter = np.random.uniform(-noise_level, noise_level, self.num_agents)
        
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        # Turn speed is inversely proportional to speed (High speed = wide turns)
        turn_rate = 0.5 / speed 
        
        self.agents[move_left, 2] -= turn_rate
        self.agents[move_right, 2] += turn_rate
        self.agents[~(move_fwd | move_left | move_right), 2] += jitter[~(move_fwd | move_left | move_right)]

        # MOVE
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # DEPOSIT
        ix, iy = self.agents[:, 0].astype(int), self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        # ATTRACTORS
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 5.0 

        # DECAY
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 5. STATE MANAGEMENT ---
if 'engine_v15' not in st.session_state:
    st.session_state.engine_v15 = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]]
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 6. SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🎛️ CONTROL PLANE")
is_running = st.sidebar.toggle("🟢 SYSTEM ONLINE", value=True)

# 1. SCENARIO
st.sidebar.markdown("#### 1. SCENARIO CONFIG")
preset = st.sidebar.selectbox("Region Topology", ["Diamond (Regional)", "Pentagon Ring", "Grid (Urban)", "Hub-Spoke (Enterprise)"])
if st.sidebar.button("⚠️ LOAD TOPOLOGY"):
    st.session_state.engine_v15 = None
    st.session_state.history = []
    if preset == "Diamond (Regional)":
        st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]]
    elif preset == "Pentagon Ring":
        c, r = (150, 150), 90
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        st.session_state.nodes = [[c[0] + r*np.cos(a), c[1] + r*np.sin(a)] for a in angles]
    elif preset == "Grid (Urban)":
        st.session_state.nodes = [[x, y] for x in range(80, 280, 70) for y in range(80, 280, 70)]
    elif preset == "Hub-Spoke (Enterprise)":
        nodes = [[150, 150]]
        nodes.extend([[150 + 110*np.cos(a), 150 + 110*np.sin(a)] for a in np.linspace(0, 2*np.pi, 7)[:-1]])
        st.session_state.nodes = nodes
    st.rerun()

st.sidebar.markdown("---")

# 2. BUSINESS
st.sidebar.markdown("#### 2. BUSINESS LOGIC")
capex_pref = st.sidebar.slider("CAPEX Limit (Decay)", 0.0, 1.0, 0.8, help="Budget Constraint. High = Minimal Cabling. Low = Mesh Network.")
decay = 0.90 + (0.09 * (1.0 - capex_pref))
redundancy_pref = st.sidebar.slider("Failover Risk (Angle)", 0.1, 1.5, 0.7, help="Risk Tolerance. High = Build Backup Loops. Low = Single Point of Failure OK.")
traffic_load = st.sidebar.slider("Projected Load (Agents)", 1000, 10000, 5000, help="Simulated Network Congestion.")

st.sidebar.markdown("---")

# 3. PHYSICS
st.sidebar.markdown("#### 3. PHYSICS IMPACT")
latency_pref = st.sidebar.slider("Transmission Speed (C)", 1.0, 5.0, 2.0, help="Simulates Signal Propagation Speed. High = HFT/Direct Lines. Low = Standard Fiber.")

# 4. ENVIRONMENT
st.sidebar.markdown("#### 4. ENVIRONMENTAL FACTORS")
terrain_diff = st.sidebar.slider("Signal Interference (Noise)", 0.05, 0.5, 0.1, help="Simulates physical obstacles/EM interference. High values cause erratic routing.")

# --- 7. INITIALIZE ---
if st.session_state.engine_v15 is None or st.session_state.engine_v15.num_agents != traffic_load:
    st.session_state.engine_v15 = BioEngine(300, 300, traffic_load)

engine = st.session_state.engine_v15
nodes_arr = np.array(st.session_state.nodes)

if is_running:
    for _ in range(12): 
        # Pass all 4 variable sets
        engine.step(st.session_state.nodes, latency_pref, decay, redundancy_pref, terrain_diff)

# --- 8. METRICS & ANALYSIS ---
if len(nodes_arr) > 1:
    mst_cost = minimum_spanning_tree(distance_matrix(nodes_arr, nodes_arr)).toarray().sum()
else:
    mst_cost = 0

cable_volume = np.sum(engine.trail_map > 1.0) / 10.0
capex_efficiency = min(100, (mst_cost / (cable_volume + 1)) * 100)

# --- 9. DASHBOARD UI ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("### 🕸️ NEUROMORPHIC ARCHITECT")
    st.caption(f"OPTIMIZATION TARGET: STEINER TREE APPROXIMATION | MODE: v15 FULL-SPECTRUM")

# DYNAMIC ANALYST REPORT
report_color = "#00FF41" if capex_efficiency > 50 else "#FFA500"
latency_status = "ULTRA-LOW (HFT)" if latency_pref > 3.0 else "STANDARD"
terrain_status = "HOSTILE" if terrain_diff > 0.3 else "STABLE"

st.markdown(f"""
<div style="border: 1px solid #333; padding: 10px; border-radius: 5px; background-color: #0A0A0A; margin-bottom: 10px;">
    <span style="color: #888; font-family: monospace; font-size: 12px;">ANALYST SUMMARY:</span><br>
    <span style="color: {report_color}; font-family: monospace; font-size: 14px;">
    > <b>{latency_status} LATENCY</b> protocol active. Routes prioritized for straight-line speed.<br>
    > <b>{terrain_status} TERRAIN</b> detected. Signal noise requires signal boosting (thicker cabling).<br>
    > Net Result: <b>{int(capex_efficiency)}% CAPEX Efficiency</b>. { "Deployment Approved." if capex_efficiency > 60 else "Review Budget."}
    </span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("DATA CENTERS", f"{len(nodes_arr)}")
m2.metric("MINIMUM VIABLE (MST)", f"{int(mst_cost)} km")
m3.metric("PROPOSED FIBER", f"{int(cable_volume)} km", delta=f"{int(cable_volume - mst_cost)} redundant", delta_color="inverse")
m4.metric("PHYSICS C", f"{latency_pref}x")
m5.metric("CAPEX SAVINGS", f"{int(capex_efficiency)}%", "vs. Mesh")

# --- 10. VISUALIZATION TRIFECTA ---
col_vis1, col_vis2, col_stats = st.columns([1, 1, 1.2])

# 1. BIOLOGICAL SOLVER
with col_vis1:
    st.markdown("**1. LATENCY TERRAIN (Solver)**")
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5)) 
    disp_map = np.log1p(engine.trail_map)
    ax1.imshow(disp_map, cmap='magma', origin='upper') 
    if len(nodes_arr) > 0:
        ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='white', s=25, edgecolors='cyan', zorder=10)
    ax1.axis('off')
    fig1.tight_layout(pad=0)
    st.pyplot(fig1, use_container_width=True)
    st.caption("Heatmap shows packet congestion. High Latency Priority forces straighter, hotter paths.")

# 2. SCHEMATIC
with col_vis2:
    st.markdown("**2. NETWORK BLUEPRINT (Output)**")
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
    ax2.set_xlim(0, 300); ax2.set_ylim(300, 0)
    
    y_weak, x_weak = np.where((engine.trail_map > 1.0) & (engine.trail_map < 3.0))
    if len(x_weak) > 0:
        ax2.scatter(x_weak, y_weak, c='#0055FF', s=0.2, alpha=0.3, label="Redundancy")
        
    y_core, x_core = np.where(engine.trail_map >= 3.0)
    if len(x_core) > 0:
        ax2.scatter(x_core, y_core, c='#00FF41', s=0.8, alpha=0.9, label="Backbone")
    
    if len(nodes_arr) > 0:
        ax2.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00FF41', s=50, marker='s', edgecolors='white', zorder=10)
        for i, (nx, ny) in enumerate(nodes_arr):
            ax2.text(nx+5, ny-5, f"DC-{i+1}", color='white', fontsize=6, fontfamily='monospace')

    ax2.axis('off')
    fig2.tight_layout(pad=0)
    st.pyplot(fig2, use_container_width=True)
    st.caption("Green = Core Backbone. Blue = Failover. Terrain difficulty adds jitter to these paths.")

# 3. TELEMETRY
with col_stats:
    st.markdown("**3. COST CONVERGENCE**")
    st.session_state.history.append({"MST Baseline": float(mst_cost), "Bio-Solver": float(cable_volume)})
    if len(st.session_state.history) > 80: st.session_state.history.pop(0)
    
    chart_data = pd.DataFrame(st.session_state.history)
    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
    
    if not chart_data.empty:
        ax3.plot(chart_data["MST Baseline"], color='#444444', linestyle='--', linewidth=1, label="Optimal (MST)")
        ax3.plot(chart_data["Bio-Solver"], color='#00FF41', linewidth=1.5, label="Actual Cost")
        
    ax3.grid(color='#222', linestyle='-', linewidth=0.5)
    ax3.spines['bottom'].set_color('#444')
    ax3.spines['left'].set_color('#444')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(frameon=False, labelcolor='#888', fontsize=8, loc='upper right')
    
    st.pyplot(fig3, use_container_width=True)
    st.caption("Convergence tracking. Note how High Latency/Terrain settings raise the 'Actual Cost' curve.")

if is_running:
    time.sleep(0.01)
    st.rerun()
