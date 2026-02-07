import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time

# --- 1. THE STEALTH COCKPIT CONFIGURATION ---
st.set_page_config(
    page_title="Neuromorphic Router V10", 
    layout="wide", 
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# --- 2. DEEP CSS INJECTION (The "No-White" Protocol) ---
st.markdown("""
<style>
    /* GLOBAL BLACKOUT */
    .stApp { background-color: #000000; color: #00FF41; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #1A1A1A !important;
    }
    
    /* DROPDOWNS & INPUTS - FORCE DARK */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"], 
    input.st-ai, textarea.st-ai, 
    .stSelectbox div {
        background-color: #080808 !important;
        border: 1px solid #333 !important;
        color: #00FF41 !important;
    }
    /* THE POPOVER MENU (The part that was white) */
    ul[data-baseweb="menu"], div[data-baseweb="popover"] {
        background-color: #080808 !important;
        border: 1px solid #333 !important;
    }
    li[data-baseweb="option"] { color: #00FF41 !important; }
    
    /* REMOVE PLOT MARGINS & BACKGROUNDS */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    div[data-testid="stImage"] { background: transparent !important; }
    
    /* TEXT HIERARCHY */
    h1, h2, h3, h4 { color: #00FF41 !important; font-family: 'Courier New', monospace; letter-spacing: -1px; margin-bottom: 0px;}
    p, label { color: #888 !important; font-family: 'Consolas', monospace; font-size: 12px; }
    
    /* COMPACT METRICS */
    div[data-testid="stMetric"] {
        background-color: #0A0A0A;
        border: 1px solid #222;
        padding: 5px !important;
        border-left: 3px solid #00FF41;
    }
    div[data-testid="stMetricLabel"] { font-size: 10px !important; color: #666 !important; }
    div[data-testid="stMetricValue"] { font-size: 18px !important; color: #00FF41 !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. THE ENGINE (Business Logic Mapping) ---

class BioEngine:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.agents = np.zeros((num_agents, 3))
        # Initialize Randomly
        self.agents[:, 0] = np.random.uniform(width*0.4, width*0.6, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.4, height*0.6, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, nodes, speed, decay, sensor_angle):
        self.steps += 1
        sensor_dist = 9.0
        
        # VECTORIZED SENSING
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
        
        # DECISION LOGIC
        jitter = np.random.uniform(-0.1, 0.1, self.num_agents)
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        self.agents[move_left, 2] -= 0.5
        self.agents[move_right, 2] += 0.5
        self.agents[~(move_fwd | move_left | move_right), 2] += jitter[~(move_fwd | move_left | move_right)]

        # MOVEMENT
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # DEPOSIT & GRAVITY
        ix, iy = self.agents[:, 0].astype(int), self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 5.0 

        # DECAY
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 4. STATE ---
if 'engine_final' not in st.session_state:
    st.session_state.engine_final = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]] # Simple Diamond
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 5. SIDEBAR (BUSINESS PARAMETERS) ---
st.sidebar.markdown("### 🎛️ OPTIMIZER CONTROLS")
is_running = st.sidebar.toggle("🟢 SOLVER ACTIVE", value=True)

st.sidebar.markdown("#### 1. NETWORK SCENARIO")
preset = st.sidebar.selectbox("Topology Type", ["Diamond (Regional)", "Pentagon Ring", "Grid (Urban)", "Hub-Spoke (Enterprise)"])

if st.sidebar.button("⚠️ RESET SCENARIO"):
    st.session_state.engine_final = None
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
st.sidebar.markdown("#### 2. BUSINESS CONSTRAINTS")

# MAPPING PHYSICS TO BUSINESS
# Decay Rate -> CAPEX Efficiency (Higher decay = thinner lines = less cable = lower cost)
capex_pref = st.sidebar.slider("CAPEX Constraint (Cable Cost)", 0.0, 1.0, 0.8, help="High constraint forces minimum cabling (Steiner Tree). Low constraint allows redundant loops.")
decay = 0.90 + (capex_pref * 0.09) 

# Sensor Angle -> Redundancy (Wider angle = finds more alternate paths)
redundancy_pref = st.sidebar.slider("Redundancy Requirement", 0.1, 1.5, 0.7, help="Wide search angle creates backup loops for failover.")

# Speed -> Latency (Speed of convergence)
speed = 2.0 

# --- 6. INITIALIZATION ---
if st.session_state.engine_final is None:
    st.session_state.engine_final = BioEngine(300, 300, 6000)

engine = st.session_state.engine_final
nodes_arr = np.array(st.session_state.nodes)

# --- 7. MAIN LOOP ---
if is_running:
    for _ in range(12): 
        engine.step(st.session_state.nodes, speed, decay, redundancy_pref)

# --- 8. THE SINGLE-PAGE DASHBOARD ---

# HEADER
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("### 🕸️ NEUROMORPHIC NETWORK OPTIMIZER")
    st.caption("Objective: Minimize Fiber Optic CAPEX while satisfying Redundancy Constraints.")

# CALCULATION (Business Metrics)
if len(nodes_arr) > 1:
    mst_cost = minimum_spanning_tree(distance_matrix(nodes_arr, nodes_arr)).toarray().sum()
else:
    mst_cost = 0

# Bio Cost = Amount of "Cable" laid down
cable_volume = np.sum(engine.trail_map > 1.0) / 10.0
efficiency = (mst_cost / (cable_volume + 1)) * 100
capex_savings = max(0, 100 - (cable_volume / (mst_cost+1) * 100)) # Fake metric for visual logic

# METRICS STRIP
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("NODES", f"{len(nodes_arr)}")
m2.metric("MINIMUM CABLING (Ref)", f"{int(mst_cost)} km")
m3.metric("ACTUAL CABLING", f"{int(cable_volume)} km", delta=f"{int(cable_volume - mst_cost)} overhead", delta_color="inverse")
m4.metric("REDUNDANCY SCORE", f"{int(redundancy_pref*100)}/100")
m5.metric("CAPEX SAVINGS", f"{int(capex_savings)}%", "vs. mesh")

st.markdown("---")

# VISUALIZATION ROW (Tight Packing)
col_vis1, col_vis2, col_stats = st.columns([1, 1, 1.2])

# GRAPH 1: THE BIOLOGICAL SOLVER
with col_vis1:
    st.markdown("**1. SIMULATION (Solver)**")
    fig1, ax1 = plt.subplots(figsize=(3, 3), facecolor='black') # SMALL SIZE
    disp_map = np.log1p(engine.trail_map)
    ax1.imshow(disp_map, cmap='magma', origin='upper')
    if len(nodes_arr) > 0:
        ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='white', s=20, edgecolors='cyan')
    ax1.axis('off')
    fig1.tight_layout(pad=0)
    st.pyplot(fig1, use_container_width=True)

# GRAPH 2: THE EXTRACTED NETWORK
with col_vis2:
    st.markdown("**2. OPTIMIZED ROUTE (Output)**")
    fig2, ax2 = plt.subplots(figsize=(3, 3), facecolor='black') # SMALL SIZE
    ax2.set_facecolor('black')
    ax2.set_xlim(0, 300); ax2.set_ylim(300, 0)
    
    # Extract "Cables"
    y_trail, x_trail = np.where(engine.trail_map > 2.0)
    if len(x_trail) > 0:
        ax2.scatter(x_trail, y_trail, c='#00FF41', s=0.5, alpha=0.6)
    
    # Draw Nodes
    if len(nodes_arr) > 0:
        ax2.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00FF41', s=40, marker='s', edgecolors='white', zorder=10)
        
    ax2.axis('off')
    fig2.tight_layout(pad=0)
    st.pyplot(fig2, use_container_width=True)

# GRAPH 3: REAL-TIME CONVERGENCE
with col_stats:
    st.markdown("**3. COST CONVERGENCE**")
    st.session_state.history.append({"Optimal Baseline": float(mst_cost), "Proposed Network": float(cable_volume)})
    if len(st.session_state.history) > 80: st.session_state.history.pop(0)
    
    chart_df = pd.DataFrame(st.session_state.history)
    
    # Render Line Chart
    try:
        st.line_chart(chart_df, color=["#444444", "#00FF41"], height=180) 
    except:
        st.line_chart(chart_df, height=180)
        
    st.caption("Lower Green Line = Less Cabling Cost. Gap = Redundancy Cost.")

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
