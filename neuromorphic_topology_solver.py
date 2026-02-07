import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import io
import pandas as pd
import time
import networkx as nx

# --- 1. CONFIGURATION & STEALTH CSS ---
st.set_page_config(
    page_title="Neuromorphic Topology Engine V5", 
    layout="wide", 
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# FORCE DARK MODE & REMOVE WHITE ELEMENTS
st.markdown("""
<style>
    /* MAIN BACKGROUND */
    .stApp { background-color: #050505; color: #a0a0a0; }
    
    /* INPUTS & DROPDOWNS - CYAN & BLACK */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"], 
    input.st-ai, 
    textarea.st-ai {
        background-color: #0a1014 !important;
        border: 1px solid #1e3a4a !important;
        color: #00E5FF !important;
    }
    
    /* REMOVE WHITE FROM DROPDOWN OPTIONS */
    ul[data-baseweb="menu"] { background-color: #0a1014 !important; border: 1px solid #00E5FF; }
    li[data-baseweb="option"] { color: #00E5FF !important; }

    /* STEALTH BUTTONS */
    .stButton>button, .stDownloadButton>button {
        color: #00E5FF !important;
        border: 1px solid #1e3a4a !important;
        background-color: #080c10 !important;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: #00E5FF !important;
        background-color: rgba(0, 229, 255, 0.15) !important;
        color: #FFFFFF !important;
    }

    /* METRIC BOXES */
    div[data-testid="stMetric"] {
        background-color: #080808;
        border: 1px solid #222;
        padding: 10px;
        border-left: 4px solid #00E5FF;
    }
    label[data-testid="stMetricLabel"] { color: #555; font-size: 11px; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #00E5FF; }

    /* TEXT HEADERS */
    h1, h2, h3 { color: #00E5FF !important; font-family: 'Courier New', monospace; text-transform: uppercase; letter-spacing: 1px;}
    
    /* PLOT BORDERS */
    .plot-container { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS ENGINE ---

def calculate_mst_cost(nodes):
    if len(nodes) < 2: return 0.0
    dist_mat = distance_matrix(nodes, nodes)
    mst = minimum_spanning_tree(dist_mat)
    return mst.toarray().sum()

class PhysarumEngine:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        
        # Initialize in a "Big Bang" center cluster for organic expansion
        self.agents = np.zeros((num_agents, 3))
        self.agents[:, 0] = np.random.uniform(width*0.4, width*0.6, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.4, height*0.6, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, sensor_angle, sensor_dist, turn_speed, speed, decay, nodes):
        self.steps += 1
        
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
        
        # DECISION
        jitter = np.random.uniform(-0.1, 0.1, self.num_agents)
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        new_angles = angles.copy()
        new_angles[move_left] -= turn_speed
        new_angles[move_right] += turn_speed
        mask_random = ~(move_fwd | move_left | move_right)
        new_angles[mask_random] += jitter[mask_random] * 5 
        
        self.agents[:, 2] = new_angles

        # MOVEMENT
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # DEPOSIT
        ix, iy = self.agents[:, 0].astype(int), self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        # ATTRACTION (FOOD)
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 2.0

        # DECAY
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. SESSION STATE ---

if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = np.random.randint(20, 280, size=(6, 2)).tolist()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. SIDEBAR ---

st.sidebar.markdown("### 💠 SYSTEM CONTROLS")
is_running = st.sidebar.toggle("RUN SIMULATION", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🗺️ Topology Scenario")
preset = st.sidebar.selectbox("Network Pattern:", ["Random Scatter", "Pentagon Ring", "Grid Lattice", "Star Hub"])
    
if st.sidebar.button("⚠️ REBOOT SYSTEM"):
    st.session_state.sim = None
    st.session_state.history = []
    if preset == "Random Scatter":
        st.session_state.nodes = np.random.randint(20, 280, size=(np.random.randint(5, 9), 2)).tolist()
    elif preset == "Pentagon Ring":
        c, r = (150, 150), 100
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        st.session_state.nodes = [[c[0] + r*np.cos(a), c[1] + r*np.sin(a)] for a in angles]
    elif preset == "Grid Lattice":
        st.session_state.nodes = [[x, y] for x in range(80, 280, 70) for y in range(80, 280, 70)]
    elif preset == "Star Hub":
        nodes = [[150, 150]]
        nodes.extend([[150 + 120*np.cos(a), 150 + 120*np.sin(a)] for a in np.linspace(0, 2*np.pi, 7)[:-1]])
        st.session_state.nodes = nodes
    st.rerun()

with st.sidebar.expander("⚙️ Physics Parameters"):
    agent_count = st.slider("Particle Flux", 1000, 10000, 5000)
    decay_rate = st.slider("Entropy Decay", 0.90, 0.99, 0.95)
    speed = st.slider("Propagation C", 1.0, 5.0, 2.0)

# --- 5. INITIALIZE ---

if st.session_state.sim is None or st.session_state.sim.num_agents != agent_count:
    st.session_state.sim = PhysarumEngine(300, 300, agent_count)

engine = st.session_state.sim
nodes_arr = np.array(st.session_state.nodes)

# --- 6. SIMULATION LOOP ---

if is_running:
    for _ in range(12):
        engine.step(0.7, 9, 0.5, speed, decay_rate, st.session_state.nodes)

# METRICS
mst_cost = calculate_mst_cost(nodes_arr)
bio_mask = engine.trail_map > 1.0
bio_cost = np.sum(bio_mask) / 10.0
st.session_state.history.append({"MST": mst_cost, "BIO": bio_cost, "STEP": engine.steps})
if len(st.session_state.history) > 150: st.session_state.history.pop(0)

# --- 7. DASHBOARD UI ---

st.title("NEUROMORPHIC TOPOLOGY SOLVER")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("NODES", f"{len(st.session_state.nodes)}")
m2.metric("EPOCH", f"{engine.steps}")
m3.metric("MST BASELINE", f"{int(mst_cost)}")
m4.metric("BIO-COST", f"{int(bio_cost)}", delta=f"{int(mst_cost - bio_cost)}")

st.markdown("---")

# SIDE-BY-SIDE LAYOUT (1:1 Ratio)
col_left, col_right = st.columns([1, 1])

# --- LEFT: THE VISUAL SOLUTION (MAP) ---
with col_left:
    st.markdown("###### 👁️ GEODESIC FLOW MAP")
    
    fig_map, ax_map = plt.subplots(figsize=(5, 4), facecolor='#050505')
    
    # 1. Base Slime Map
    disp_map = np.log1p(engine.trail_map)
    ax_map.imshow(disp_map, cmap='winter', origin='upper', aspect='equal')
    
    # 2. Nodes
    if len(nodes_arr) > 0:
        ax_map.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00E5FF', s=80, edgecolors='white', linewidth=1.0, zorder=10)
    
    # 3. SOLUTION OVERLAY (Steiner Approximation)
    # Highlight high-traffic paths as "Cables"
    if st.toggle("Post-Process: Extract Graph Solution", value=False):
        # Threshold the map to find strong connections
        rows, cols = np.where(engine.trail_map > 2.0)
        ax_map.scatter(cols, rows, s=0.5, c='#FFFF00', alpha=0.1) # Yellow dust for cables
        ax_map.set_title("Steiner Tree Approximation", color='#FFFF00', fontsize=8)

    ax_map.axis('off')
    fig_map.tight_layout(pad=0)
    st.pyplot(fig_map, use_container_width=True)


# --- RIGHT: THE TELEMETRY PROOF (GRAPH) ---
with col_right:
    st.markdown("###### 📉 CONVERGENCE TELEMETRY")
    
    # CUSTOM MATPLOTLIB GRAPH (To remove white backgrounds)
    hist_df = pd.DataFrame(st.session_state.history)
    
    fig_chart, ax_chart = plt.subplots(figsize=(5, 3), facecolor='#050505') # Dark BG
    ax_chart.set_facecolor('#050505') # Dark Plot Area
    
    if not hist_df.empty:
        # Plot MST (Baseline)
        ax_chart.plot(hist_df['STEP'], hist_df['MST'], color='#444444', linestyle='--', linewidth=1, label="MST (Optimal)")
        # Plot Bio (Actual)
        ax_chart.plot(hist_df['STEP'], hist_df['BIO'], color='#00E5FF', linewidth=1.5, label="Bio-Solver")
        
        # Styling
        ax_chart.grid(color='#222222', linestyle='-', linewidth=0.5)
        ax_chart.spines['bottom'].set_color('#444444')
        ax_chart.spines['left'].set_color('#444444')
        ax_chart.spines['top'].set_visible(False)
        ax_chart.spines['right'].set_visible(False)
        ax_chart.tick_params(axis='x', colors='#666666', labelsize=8)
        ax_chart.tick_params(axis='y', colors='#666666', labelsize=8)
        
        # Legend
        leg = ax_chart.legend(loc='upper right', facecolor='#050505', edgecolor='#333333', fontsize=8)
        for text in leg.get_texts():
            text.set_color("#888888")

    st.pyplot(fig_chart, use_container_width=True)
    
    # EXPORT CONTROLS (Grouped)
    c1, c2 = st.columns(2)
    with c1:
        df = pd.DataFrame(st.session_state.nodes, columns=["X", "Y"])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 DATA EXPORT", csv, "topology.csv", "text/csv", use_container_width=True)
    with c2:
        img_buf = io.BytesIO()
        fig_map.savefig(img_buf, format='png', facecolor='#050505', bbox_inches='tight', pad_inches=0)
        st.download_button("📸 SNAPSHOT", img_buf.getvalue(), "network_state.png", "image/png", use_container_width=True)

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
