import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import io
import pandas as pd
import time
import json

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Neuromorphic Topology Engine V4", 
    layout="wide", 
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# Deep Stealth / Cyber-Security CSS
st.markdown("""
<style>
    /* 1. MAIN BACKGROUND & TEXT */
    .stApp { background-color: #050505; color: #a0a0a0; }
    
    /* 2. ELIMINATE ALL WHITE BOXES (Dropdowns, Inputs) */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"], 
    input.st-ai, 
    textarea.st-ai {
        background-color: #0a1014 !important;
        border: 1px solid #333 !important;
        color: #00E5FF !important;
    }
    div[data-baseweb="select"] > div:hover { border-color: #00E5FF !important; }
    
    /* Dropdown Options (The pop-up list) */
    ul[data-baseweb="menu"] { background-color: #0a1014 !important; border: 1px solid #333; }
    li[data-baseweb="option"] { color: #00E5FF !important; }
    
    /* 3. BUTTONS (Standard & Download) - STEALTH DEFAULT */
    .stButton>button, .stDownloadButton>button {
        color: #00E5FF !important;
        border: 1px solid #1e3a4a !important;
        background-color: #080c10 !important; /* DARK DEFAULT */
        font-family: 'Courier New', monospace;
        font-size: 12px;
        transition: all 0.2s ease-in-out;
    }
    /* HOVER STATE */
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: #00E5FF !important;
        background-color: rgba(0, 229, 255, 0.1) !important;
        box-shadow: 0 0 8px rgba(0, 229, 255, 0.4);
        color: #FFFFFF !important;
    }

    /* 4. METRICS & ALERTS */
    div[data-testid="stMetric"] {
        background-color: #080808;
        border: 1px solid #222;
        padding: 5px;
        border-left: 3px solid #00E5FF;
    }
    label[data-testid="stMetricLabel"] { color: #666; font-size: 11px; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #00E5FF; text-shadow: 0 0 5px rgba(0,229,255,0.5); }
    
    /* 5. HEADERS & TEXT */
    h1, h2, h3 { color: #00E5FF !important; font-family: 'Courier New', monospace; letter-spacing: 2px; text-transform: uppercase; }
    p, li { font-family: 'Consolas', monospace; font-size: 14px; color: #888; }
    
    /* 6. SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #020202; border-right: 1px solid #111; }
    
    /* 7. RADIO BUTTONS */
    div[role="radiogroup"] label { color: #888 !important; }
    div[role="radiogroup"] label:hover { color: #00E5FF !important; }

</style>
""", unsafe_allow_html=True)

# --- 2. MATH & PHYSICS KERNEL ---

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
        
        # Initialization: Ring Formation (More visually interesting than random)
        self.agents = np.zeros((num_agents, 3))
        thetas = np.random.uniform(0, 2*np.pi, num_agents)
        r = np.random.uniform(0, 10, num_agents) # Start at center
        self.agents[:, 0] = width/2 + r * np.cos(thetas)
        self.agents[:, 1] = height/2 + r * np.sin(thetas)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, sensor_angle, sensor_dist, turn_speed, speed, decay, nodes):
        self.steps += 1
        
        # 1. Sensing
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
        
        # 2. Steering
        jitter = np.random.uniform(-0.15, 0.15, self.num_agents)
        
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        new_angles = angles.copy()
        new_angles[move_left] -= turn_speed
        new_angles[move_right] += turn_speed
        mask_random = ~(move_fwd | move_left | move_right)
        new_angles[mask_random] += jitter[mask_random] * 4
        
        self.agents[:, 2] = new_angles

        # 3. Movement
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # 4. Deposit
        ix, iy = self.agents[:, 0].astype(int), self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        # 5. Node Gravity
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 2.5

        # 6. Decay
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. SESSION MANAGEMENT ---

if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = np.random.randint(20, 280, size=(6, 2)).tolist()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. SIDEBAR CONTROLS ---

st.sidebar.markdown("### 💠 KERNEL ACCESS")
is_running = st.sidebar.toggle("System Active", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🗺️ Topology Config")
preset = st.sidebar.selectbox("Load Protocol:", ["Random Scatter", "Pentagon Ring", "Grid Lattice", "Star Hub"])
    
if st.sidebar.button("INITIALIZE PROTOCOL"):
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

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Physics Parameters"):
    agent_count = st.slider("Particle Flux", 1000, 10000, 5000)
    decay_rate = st.slider("Entropy Decay", 0.90, 0.99, 0.95)
    speed = st.slider("Propagation C", 1.0, 5.0, 2.0)

# --- 5. INITIALIZATION ---

if st.session_state.sim is None or st.session_state.sim.num_agents != agent_count:
    st.session_state.sim = PhysarumEngine(300, 300, agent_count)

engine = st.session_state.sim
nodes_arr = np.array(st.session_state.nodes)

# --- 6. SIMULATION LOOP & RENDER ---

if is_running:
    for _ in range(12):
        engine.step(0.7, 9, 0.5, speed, decay_rate, st.session_state.nodes)

# Metrics
mst_cost = calculate_mst_cost(nodes_arr)
bio_mask = engine.trail_map > 1.0
bio_cost = np.sum(bio_mask) / 10.0
st.session_state.history.append({"MST": mst_cost, "BIO": bio_cost})
if len(st.session_state.history) > 100: st.session_state.history.pop(0)

# LAYOUT: COMPACT GRID
st.title("NEUROMORPHIC TOPOLOGY SOLVER")

# Top Metrics Bar
m1, m2, m3, m4 = st.columns(4)
m1.metric("NODES", f"{len(st.session_state.nodes)}")
m2.metric("EPOCH", f"{engine.steps}")
m3.metric("MST BASELINE", f"{int(mst_cost)}")
m4.metric("BIO-COST", f"{int(bio_cost)}", delta=f"{int(mst_cost - bio_cost)}")

st.markdown("---")

# Main Visuals (Split 1.5 : 1 for a smaller map)
col_vis, col_stats = st.columns([1.5, 1])

with col_vis:
    # Plotting - SMALLER FIGURE SIZE (4x4 inches)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#050505')
    
    # Data Normalization for visuals
    disp_map = np.log1p(engine.trail_map)
    
    # "Cyber" Color Scheme - Winter/Cool
    ax.imshow(disp_map, cmap='winter', origin='upper', aspect='equal')
    
    # Nodes
    if len(nodes_arr) > 0:
        ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00E5FF', s=80, edgecolors='white', linewidth=1.0, zorder=10)
    
    ax.axis('off')
    # Tight layout to remove whitespace
    fig.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)

with col_stats:
    st.markdown("###### 📉 CONVERGENCE TELEMETRY")
    
    # Live Line Chart
    chart_data = pd.DataFrame(st.session_state.history)
    st.line_chart(chart_data, color=["#444444", "#00E5FF"], height=180)
    
    st.markdown("###### 💾 DATA EXPORT")
    
    # Button 1: CSV
    df = pd.DataFrame(st.session_state.nodes, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("DOWNLOAD NODE TELEMETRY", csv, "topology.csv", "text/csv", use_container_width=True)
    
    # Button 2: Image
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', facecolor='#050505', bbox_inches='tight', pad_inches=0)
    st.download_button("CAPTURE VISUAL STATE", img_buf.getvalue(), "network_state.png", "image/png", use_container_width=True)

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
