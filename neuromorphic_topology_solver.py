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
    page_title="Neuromorphic Topology Engine V3", 
    layout="wide", 
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# Cyberpunk/Security Analyst Theme
st.markdown("""
<style>
    /* Global Reset */
    .stApp { background-color: #020406; color: #C0C0C0; }
    
    /* Compact Metrics */
    div[data-testid="stMetric"] {
        background-color: #0a1014;
        border: 1px solid #1e2a36;
        padding: 5px 10px;
        border-radius: 2px;
        border-left: 2px solid #00E5FF;
    }
    label[data-testid="stMetricLabel"] { font-size: 12px; }
    div[data-testid="stMetricValue"] { font-size: 18px; color: #00E5FF; }

    /* Headers */
    h1, h2, h3 { color: #00E5FF !important; font-family: 'Courier New', monospace; letter-spacing: -1px; text-transform: uppercase;}
    
    /* Buttons & Inputs */
    .stTextArea textarea { background-color: #0a1014; color: #00E5FF; border: 1px solid #333; }
    .stButton>button {
        color: #00E5FF;
        border: 1px solid #00E5FF;
        background: rgba(0, 229, 255, 0.05);
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    .stButton>button:hover {
        background: #00E5FF;
        color: #000;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
    }
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
        
        # Agents: [x, y, angle] initialized in a "Big Bang" center cluster
        self.agents = np.zeros((num_agents, 3))
        self.agents[:, 0] = np.random.uniform(width*0.45, width*0.55, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.45, height*0.55, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, sensor_angle, sensor_dist, turn_speed, speed, decay, nodes):
        self.steps += 1
        
        # 1. Sensing (Vectorized Geodesic Search)
        angles = self.agents[:, 2]
        
        # Helper to wrap coordinates (Toroidal topology)
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
        
        # 2. Steering Logic (Flux Maximization)
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

        # 3. Movement & Deposition
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        ix, iy = self.agents[:, 0].astype(int), self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 1.0) 
        
        # 4. Node Gravity (Attractors)
        for sx, sy in nodes:
            y_min, y_max = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_min, x_max = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_min:y_max, x_min:x_max] += 2.0

        # 5. Decay (Entropy)
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. SESSION MANAGEMENT ---

if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = np.random.randint(20, 280, size=(6, 2)).tolist()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. SIDEBAR CONTROLS ---

st.sidebar.markdown("## 💠 SYSTEM KERNEL")
is_running = st.sidebar.toggle("RUN SIMULATION", value=True)

with st.sidebar.expander("🗺️ Topology Editor", expanded=True):
    preset = st.selectbox("Preset Patterns:", ["Random Scatter", "Pentagon Ring", "Grid Lattice", "Star Hub"])
    
    if st.button("RESET / APPLY PRESET"):
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

    # Manual Override
    manual_input = st.text_area("Inject Coordinates (JSON)", value="", placeholder="[[100,100], [200,200]]")
    if st.button("INJECT COORDINATES"):
        try:
            coords = json.loads(manual_input)
            st.session_state.nodes = coords
            st.session_state.sim = None
            st.rerun()
        except:
            st.error("Invalid JSON format.")

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

# Main Visuals (Split 2:1)
col_vis, col_stats = st.columns([2, 1])

with col_vis:
    st.markdown("### 👁️ GEODESIC FLOW MAP")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#020406')
    
    # Data Normalization for visuals
    disp_map = np.log1p(engine.trail_map)
    
    # "Cyber" Color Scheme
    ax.imshow(disp_map, cmap='winter', origin='upper', aspect='equal')
    
    # Nodes
    if len(nodes_arr) > 0:
        ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='#00E5FF', s=100, edgecolors='white', linewidth=1.5, zorder=10)
    
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)

with col_stats:
    st.markdown("### 📈 CONVERGENCE TELEMETRY")
    
    # Live Line Chart
    chart_data = pd.DataFrame(st.session_state.history)
    st.line_chart(chart_data, color=["#FFFFFF", "#00E5FF"], height=200)
    
    st.info("""
    **SYSTEM STATUS:**
    * **Algorithm:** Physarum Polycephalum (Slime Mold)
    * **Objective:** Steiner Tree Approximation
    * **Latency:** < 12ms
    """)
    
    # Download
    if st.button("💾 EXPORT TOPOLOGY DATA"):
        df = pd.DataFrame(st.session_state.nodes, columns=["X", "Y"])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "topology.csv", "text/csv")

# AUTO-LOOP
if is_running:
    time.sleep(0.01)
    st.rerun()
