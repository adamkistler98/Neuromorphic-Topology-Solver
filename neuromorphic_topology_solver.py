import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import io
import pandas as pd # Needed for the CSV export

# --- 1. UI CONFIGURATION ---
st.set_page_config(
    page_title="Bio-Mimetic Router V5+", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS for Readability
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0E1117; }
    
    /* Headers - Bright Green */
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier New', monospace; }
    
    /* Body Text, Captions, List Items - BRIGHT WHITE */
    .stMarkdown p, .stCaption, li { color: #FFFFFF !important; font-size: 16px; }
    
    /* Metrics - Green Value, White Label */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-size: 24px; }
    div[data-testid="stMetricLabel"] { color: #FFFFFF !important; font-size: 14px; font-weight: bold; }
    
    /* Info Boxes - Dark Green BG, White Text */
    div[data-testid="stAlert"] { background-color: #112211; border: 1px solid #00FF41; }
    div[data-testid="stAlert"] p { color: #FFFFFF !important; }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #333; }
    
    /* Buttons */
    .stButton>button { border: 1px solid #00FF41; color: #00FF41; background-color: transparent; }
    .stButton>button:hover { background-color: rgba(0,255,65,0.2); color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE SIMULATION ENGINE ---
class SlimeEngine:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        
        # AGENTS: [x, y, angle]
        self.agents = np.zeros((num_agents, 3))
        self.agents[:, 0] = np.random.rand(num_agents) * width
        self.agents[:, 1] = np.random.rand(num_agents) * height
        self.agents[:, 2] = np.random.rand(num_agents) * 2 * np.pi
        
        # MAPS
        self.trail_map = np.zeros((height, width))
        self.steps = 0

    def step(self, sensor_angle, sensor_dist, turn_speed, speed, decay, nodes):
        self.steps += 1
        
        # --- SENSING ---
        angles = self.agents[:, 2]
        wrap = lambda x, m: (x.astype(int) % m)
        
        lx = wrap(self.agents[:, 0] + np.cos(angles - sensor_angle) * sensor_dist, self.width)
        ly = wrap(self.agents[:, 1] + np.sin(angles - sensor_angle) * sensor_dist, self.height)
        cx = wrap(self.agents[:, 0] + np.cos(angles) * sensor_dist, self.width)
        cy = wrap(self.agents[:, 1] + np.sin(angles) * sensor_dist, self.height)
        rx = wrap(self.agents[:, 0] + np.cos(angles + sensor_angle) * sensor_dist, self.width)
        ry = wrap(self.agents[:, 1] + np.sin(angles + sensor_angle) * sensor_dist, self.height)
        
        l_val = self.trail_map[ly, lx]
        c_val = self.trail_map[cy, cx]
        r_val = self.trail_map[ry, rx]
        
        # --- DECISION MAKING ---
        mask_fwd = (c_val > l_val) & (c_val > r_val)
        mask_left = (l_val > r_val) & (l_val > c_val)
        mask_right = (r_val > l_val) & (r_val > c_val)
        
        jitter = np.random.uniform(-0.2, 0.2, self.num_agents)
        self.agents[mask_left, 2] -= turn_speed
        self.agents[mask_right, 2] += turn_speed
        self.agents[~mask_fwd & ~mask_left & ~mask_right, 2] += jitter[~mask_fwd & ~mask_left & ~mask_right]

        # --- MOVEMENT ---
        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        # --- DEPOSIT PHEROMONES ---
        ix = self.agents[:, 0].astype(int)
        iy = self.agents[:, 1].astype(int)
        np.add.at(self.trail_map, (iy, ix), 5.0) 
        
        # --- NODE GRAVITY ---
        for sx, sy in nodes:
            y_min, y_max = max(0, sy-4), min(self.height, sy+4)
            x_min, x_max = max(0, sx-4), min(self.width, sx+4)
            self.trail_map[y_min:y_max, x_min:x_max] += 5.0

        # --- DECAY ---
        self.trail_map = gaussian_filter(self.trail_map, sigma=0.7) * decay

# --- 3. STATE MANAGEMENT ---
if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = [[150, 50], [250, 120], [200, 250], [100, 250], [50, 120]]

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Network Controls")

# PLAY/PAUSE TOGGLE
st.sidebar.markdown("### ⏯️ Simulation State")
is_paused = st.sidebar.toggle("⏸️ Pause Simulation", value=False)

st.sidebar.markdown("### 🧬 Bio-Parameters")
num_agents = st.sidebar.slider("Packet Load (Agents)", 1000, 10000, 5000, 500)
decay = st.sidebar.slider("Decay Rate (Optimization)", 0.80, 0.99, 0.92, 0.01)

st.sidebar.markdown("### 🚀 Physics Engine")
speed = st.sidebar.slider("Transmission Speed", 0.5, 4.0, 1.5)
sensor_angle = st.sidebar.slider("Sensor Angle", 0.1, 1.5, 0.6)
turn_speed = st.sidebar.slider("Turn Agility", 0.1, 1.0, 0.4)

st.sidebar.markdown("---")
if st.sidebar.button("⚠️ Reset Topology", type="primary"):
    st.session_state.sim = None
    st.session_state.nodes = np.random.randint(20, 280, size=(np.random.randint(4, 7), 2)).tolist()
    st.rerun()

# --- 5. INITIALIZE ENGINE ---
if st.session_state.sim is None or st.session_state.sim.num_agents != num_agents:
    st.session_state.sim = SlimeEngine(300, 300, num_agents)
engine = st.session_state.sim

# --- 6. MAIN DASHBOARD ---
st.title("BIO-MIMETIC NETWORK OPTIMIZER")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Network Nodes", f"{len(st.session_state.nodes)}")
m2.metric("Active Packets", f"{num_agents}")
m3.metric("Simulation Epoch", f"{engine.steps}")
cost = np.sum(engine.trail_map > 0.1)
m4.metric("Cabling Cost (km)", f"{int(cost)}")

st.markdown("---")

col_map, col_info = st.columns([2, 1])

with col_map:
    # ONLY RUN PHYSICS IF NOT PAUSED
    if not is_paused:
        for _ in range(10):
            engine.step(sensor_angle, 9, turn_speed, speed, decay, st.session_state.nodes)
    
    # RENDER MAP
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0E1117')
    
    # Network Trails
    disp_map = np.log1p(engine.trail_map)
    vmax = np.percentile(disp_map, 99) if np.any(disp_map) else 1
    ax.imshow(disp_map, cmap='plasma', vmin=0, vmax=vmax, origin='upper')
    
    # Servers
    nx, ny = zip(*st.session_state.nodes)
    ax.scatter(nx, ny, c='white', s=200, edgecolors='#00FFFF', linewidth=3, zorder=10)
    ax.scatter(nx, ny, c='#00FFFF', s=800, alpha=0.3, zorder=5) # Glow

    ax.axis('off')
    st.pyplot(fig)
    
    # --- NEW: DOWNLOAD BUTTONS (Placed Directly Under Image) ---
    btn1, btn2 = st.columns(2)
    
    # Button 1: Image Download
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', facecolor='#0E1117', bbox_inches='tight')
    btn1.download_button(
        label="📸 Snapshot Map",
        data=img_buf.getvalue(),
        file_name=f"network_state_{engine.steps}.png",
        mime="image/png",
        use_container_width=True
    )
    
    # Button 2: Data Export
    df_nodes = pd.DataFrame(st.session_state.nodes, columns=["X_Coord", "Y_Coord"])
    df_nodes['Efficiency_Metric'] = int(cost)
    csv = df_nodes.to_csv(index=False).encode('utf-8')
    btn2.download_button(
        label="💾 Export Telemetry",
        data=csv,
        file_name="network_data.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_info:
    st.subheader("📡 Live Status")
    
    # Live Logic Updates
    if is_paused:
        st.warning("⏸️ **Status: PAUSED**\n\nSimulation frozen. Adjust parameters and unpause to continue.")
    elif engine.steps < 50:
        st.info("🟡 **Status: EXPLORING**\n\nAgents are randomly searching the grid to discover available Data Centers.")
    elif engine.steps < 200:
        st.success("🟢 **Status: CONVERGING**\n\nPrimary trunk lines are forming. Redundant paths are being pruned.")
    else:
        st.info("🔵 **Status: OPTIMIZED**\n\nNetwork has reached steady-state efficiency (Steiner Tree approximation).")

    st.markdown("### 🗺️ Legend")
    st.markdown("""
    * **⚪ White/Cyan Dots:** Data Centers (Servers)
    * **🟡 bright Yellow Lines:** High-Bandwidth Trunks
    * **🟣 Purple/Pink Haze:** Low-Traffic / Latency
    """)
    
    st.markdown("### 🧠 Bio-Algorithm")
    st.caption("""
    This simulation replicates **Physarum polycephalum** (Slime Mold). 
    Instead of using standard pathfinding algorithms (like Dijkstra), 
    we treat data packets as biological organisms that 'eat' network availability.
    """)

# --- 7. AUTO-LOOP LOGIC ---
if not is_paused:
    # This triggers the script to re-run immediately, creating the animation loop
    time.sleep(0.01) # Tiny sleep to prevent CPU spiking
    st.rerun()