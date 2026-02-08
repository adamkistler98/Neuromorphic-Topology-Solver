import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import random

# --- 1. THE NUCLEAR OPTION: GLOBAL DARK PLOTTING ---
plt.style.use('dark_background') 

# --- 2. STEALTH CONFIGURATION ---
st.set_page_config(
    page_title="NetOpt v36: Sentient", 
    layout="wide", 
    page_icon="🧠",
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
    
    /* INPUT FIELDS - Monospace for CLI feel */
    div[data-baseweb="input"] > div {
        background-color: #0A0A0A !important;
        color: #00FF41 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace;
    }

    /* STEALTH BUTTONS */
    div.stButton > button {
        background-color: #0A0A0A !important;
        color: #00FF41 !important;
        border: 1px solid #333 !important;
    }
    div.stButton > button:hover {
        background-color: #1A1A1A !important;
        border: 1px solid #00FF41 !important;
        color: #00FF41 !important;
    }
    
    /* AGENT TERMINAL STYLE */
    .agent-terminal {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #00FF41;
        background-color: #000;
        border: 1px solid #333;
        padding: 5px;
        height: 120px;
        overflow-y: auto;
        margin-top: 10px;
        border-left: 2px solid #00FF41;
    }
    
    /* CHEAT SHEET STYLE */
    .cmd-cheat-sheet {
        font-family: 'Courier New', monospace;
        font-size: 10px;
        color: #666;
        margin-top: 5px;
        line-height: 1.4;
        border-top: 1px dashed #333;
        padding-top: 8px;
    }
    .cmd-key { color: #00FF41; font-weight: bold; }
    
    /* REMOVE ALL PLOT PADDING/MARGINS */
    .main .block-container { padding: 1rem; }
    div[data-testid="stImage"] { background: transparent !important; margin: 0px; }
    button[title="View fullscreen"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- 4. BIO-ENGINE (PHYSICS KERNEL) ---
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

# --- 5. THE COMMAND AGENT BRAIN ---
class CommandAgent:
    def __init__(self, start_capex, start_redundancy, node_complexity=1):
        self.current_params = [start_capex, start_redundancy] 
        self.best_params = [start_capex, start_redundancy]
        self.best_score = 0
        
        # INTELLIGENCE SCALING
        self.base_rate = 0.05 * (1 + (node_complexity * 0.1))
        self.learning_rate = self.base_rate
        
        self.cooldown = 0
        self.last_action_idx = 0
        self.last_change = 0
        
        self.override_mode = None 
        self.is_frozen = False
        self.fail_streak = 0 # Track local traps

    def process_command(self, cmd):
        cmd = cmd.lower()
        msg = "CMD: Unknown Command."
        
        if "freeze" in cmd or "halt" in cmd:
            self.is_frozen = True
            msg = "CMD: SYSTEM PAUSED. Optimizers Locked."
        elif "growth" in cmd or "expand" in cmd:
            self.override_mode = "GROWTH"
            self.current_params[0] = 0.8; self.current_params[1] = 0.8  
            self.is_frozen = False
            msg = "CMD: PRIORITY >> BALANCED GROWTH."
        elif "audit" in cmd or "status" in cmd:
            msg = f"AUDIT: Mode={self.override_mode or 'AUTO'} | LR={self.learning_rate:.3f} | Best={int(self.best_score)}%"
        elif "speed" in cmd or "fast" in cmd:
            self.override_mode = "SPEED"
            self.is_frozen = False
            msg = "CMD: PRIORITY >> LOW LATENCY."
        elif "cost" in cmd or "budget" in cmd:
            self.override_mode = "COST"
            self.is_frozen = False
            msg = "CMD: PRIORITY >> MINIMAL CAPEX."
        elif "stable" in cmd or "safe" in cmd or "emergency" in cmd:
            self.override_mode = "STABLE"
            self.is_frozen = False
            msg = "CMD: PRIORITY >> MAX REDUNDANCY."
        elif "reset" in cmd or "auto" in cmd:
            self.override_mode = None
            self.is_frozen = False
            msg = "CMD: MANUAL OVERRIDE CLEARED."
            
        return msg, self.current_params

    def propose_action(self, current_load, current_score):
        if self.is_frozen:
            return self.current_params, True

        if self.cooldown > 0:
            self.cooldown -= 1
            return self.current_params, True 
        
        # --- NEW: PRECISION MODE ---
        # If we are doing well (>85%), reduce learning rate for fine tuning
        if current_score > 85:
            self.learning_rate = self.base_rate * 0.2 # Fine adjustments
        else:
            self.learning_rate = self.base_rate # Big swings
            
        # --- NEW: STUCK DETECTION ---
        # If we failed 10 times, jump out of local minima
        if self.fail_streak > 10:
            idx = random.choice([0, 1])
            # Random Jump
            self.current_params[idx] = random.uniform(0.1, 0.9)
            self.fail_streak = 0
            self.cooldown = 4
            return self.current_params, False

        idx = random.choice([0, 1])
        change = random.choice([-self.learning_rate, self.learning_rate])

        # Bias the change based on Mode
        if self.override_mode == "SPEED":
            if idx == 1 and change > 0: change *= -0.5 
            if idx == 1 and self.current_params[1] > 0.4: change = -abs(change) 
        elif self.override_mode == "COST":
            if idx == 0 and change < 0: change *= -0.5 
            if idx == 0 and self.current_params[0] < 0.9: change = abs(change) 
        elif self.override_mode == "STABLE":
            if idx == 1 and change < 0: change *= -0.5 
            if idx == 1 and self.current_params[1] < 1.2: change = abs(change) 

        candidate = self.current_params.copy()
        candidate[idx] += change
        
        # Load Reaction
        if current_load > 7000 and self.override_mode is None:
             candidate[1] = max(candidate[1], 1.1) 
        
        # Clamp
        candidate[0] = max(0.01, min(0.99, candidate[0])) 
        candidate[1] = max(0.1, min(1.5, candidate[1]))  
        
        self.last_action_idx = idx
        self.last_change = change
        self.cooldown = 2 
        
        return candidate, False 

    def learn(self, efficiency_score, candidate_params):
        if self.is_frozen: return "System Paused."
        if self.cooldown > 0: return f"Processing Topology... ({self.cooldown})"
        
        msg = ""
        delta = efficiency_score - self.best_score
        
        if self.override_mode:
            self.best_score = efficiency_score
            self.best_params = candidate_params
            self.current_params = candidate_params
            return f"EXECUTING: {self.override_mode} PROTOCOLS."

        if delta > -3: # Tolerant of small drops
            if delta > 0:
                self.best_score = efficiency_score
                self.best_params = candidate_params
                self.current_params = candidate_params
                self.fail_streak = 0
                msg = f"SUCCESS: New Baseline {int(efficiency_score)}%."
            else:
                self.current_params = candidate_params
                msg = "HOLD: Exploring local minima..."
        else:
            revert = candidate_params.copy()
            revert[self.last_action_idx] -= self.last_change
            self.current_params = revert
            self.fail_streak += 1
            msg = f"FAIL: Signal loss. Reverting."
            
        return msg

# --- 6. STATE MANAGEMENT ---
if 'engine_v36' not in st.session_state:
    st.session_state.engine_v36 = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]]
if 'history' not in st.session_state:
    st.session_state.history = []
if 'agent_log' not in st.session_state:
    st.session_state.agent_log = ["Agent initialized. Awaiting feedback loop..."]

# Initialize keys
if 'capex_key' not in st.session_state: st.session_state.capex_key = 0.8
if 'redundancy_key' not in st.session_state: st.session_state.redundancy_key = 0.7

# --- 7. SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🎛️ CONTROL PLANE")
control_mode = st.sidebar.radio("Operation Mode", ["🤖 Autonomous Agent", "Manual Operator"], horizontal=True)
is_agent = (control_mode == "🤖 Autonomous Agent")

# 1. SCENARIO CUSTOMIZATION
st.sidebar.markdown("#### 1. NETWORK SCALE")
node_count = st.sidebar.slider("Number of Data Centers", 3, 15, len(st.session_state.nodes))

reshuffle = st.sidebar.button("Randomize")

if reshuffle or len(st.session_state.nodes) != node_count:
    st.session_state.engine_v36 = None
    st.session_state.history = []
    st.session_state.agent_log = [f"Network resized to {node_count} nodes. Memory wiped."]
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    
    new_nodes = []
    for _ in range(node_count):
        new_nodes.append([np.random.randint(50, 250), np.random.randint(50, 250)])
    st.session_state.nodes = new_nodes
    st.rerun()

# --- PRESET LOADER ---
preset_options = [
    "Diamond (Regional)", "Pentagon Ring", "Grid (Urban)", "Hub-Spoke (Enterprise)", 
    "Twin Cities (Dual Cluster)", "Global Link (Trans-Oceanic)", "Starlink (LEO Mesh)",
    "Tri-State (3 Clusters)", "Pipeline (Linear)", "The Void (Perimeter)"
]
preset = st.sidebar.selectbox("Load Preset", preset_options)

# BUDGET (RELOCATED)
manual_budget = st.sidebar.number_input("Target Budget (k)", min_value=0, value=250, step=10)

if st.sidebar.button("⚠️ LOAD PRESET"):
    st.session_state.engine_v36 = None
    st.session_state.history = []
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    
    # Preset Logic
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
    elif preset == "Twin Cities (Dual Cluster)":
        c1 = [[np.random.randint(50, 110), np.random.randint(50, 110)] for _ in range(4)]
        c2 = [[np.random.randint(190, 250), np.random.randint(190, 250)] for _ in range(4)]
        st.session_state.nodes = c1 + c2
    elif preset == "Global Link (Trans-Oceanic)":
        left = [[50, y] for y in np.linspace(50, 250, 5)]
        right = [[250, y] for y in np.linspace(50, 250, 5)]
        st.session_state.nodes = left + right
    elif preset == "Starlink (LEO Mesh)":
        st.session_state.nodes = [[np.random.randint(20, 280), np.random.randint(20, 280)] for _ in range(12)]
    elif preset == "Tri-State (3 Clusters)":
        c1 = [[np.random.randint(40, 100), np.random.randint(40, 100)] for _ in range(3)]
        c2 = [[np.random.randint(200, 260), np.random.randint(40, 100)] for _ in range(3)]
        c3 = [[np.random.randint(120, 180), np.random.randint(200, 260)] for _ in range(3)]
        st.session_state.nodes = c1 + c2 + c3
    elif preset == "Pipeline (Linear)":
        st.session_state.nodes = [[30 + i*20, 30 + i*20] for i in range(12)]
    elif preset == "The Void (Perimeter)":
        n, w, h = 10, 300, 300
        nodes = []
        for i in range(n):
            nodes.append([random.randint(10, w-10), random.randint(10, 40)])
            nodes.append([random.randint(10, w-10), random.randint(h-40, h-10)])
            nodes.append([random.randint(10, 40), random.randint(10, h-10)])
            nodes.append([random.randint(w-40, w-10), random.randint(10, h-10)])
        st.session_state.nodes = random.sample(nodes, 12)
        
    st.rerun()

st.sidebar.markdown("---")

# 2. AGENT LOGIC
if is_agent:
    if 'agent_brain' not in st.session_state:
        st.session_state.agent_brain = CommandAgent(st.session_state.capex_key, st.session_state.redundancy_key, len(st.session_state.nodes))
        st.session_state.agent_log.append("Agent: Command Interface Online.")
    
    st.sidebar.markdown("#### 2. AGENT SERVO CONTROL")
    st.sidebar.info(f"Optimization Active. Target > 90%.")
    
    latency_pref = 3.0
    terrain_diff = 0.1
else:
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    st.sidebar.markdown("#### 2. MANUAL OVERRIDE")
    latency_pref = st.sidebar.slider("Speed (C)", 1.0, 5.0, 2.0)
    terrain_diff = st.sidebar.slider("Noise", 0.05, 0.5, 0.1)

# 3. RENDER SLIDERS (HARD CLAMPED)
safe_capex = float(np.clip(st.session_state.capex_key, 0.0, 1.0))
safe_redundancy = float(np.clip(st.session_state.redundancy_key, 0.1, 1.5))

capex_pref = st.sidebar.slider("CAPEX Limit (Decay)", 0.0, 1.0, value=safe_capex, key="capex_slider")
redundancy_pref = st.sidebar.slider("Failover Risk (Angle)", 0.1, 1.5, value=safe_redundancy, key="redundancy_slider")
traffic_load = st.sidebar.slider("Load (Agents)", 1000, 8000, 5000)

# Sync sliders
if not is_agent:
    st.session_state.capex_key = capex_pref
    st.session_state.redundancy_key = redundancy_pref

# 4. COMMAND CONSOLE
if is_agent:
    st.sidebar.markdown("---")
    
    with st.sidebar.form(key='cli_form', clear_on_submit=True):
        user_cmd = st.text_input("ENTER COMMAND >_", placeholder="Type 'HELP' for list")
        submit_btn = st.form_submit_button("EXECUTE")
    
    if submit_btn and user_cmd:
        msg, new_params = st.session_state.agent_brain.process_command(user_cmd)
        st.session_state.capex_key = new_params[0]
        st.session_state.redundancy_key = new_params[1]
        st.session_state.agent_log.append(msg)
        st.rerun()

    st.sidebar.markdown("""
    <div class="cmd-cheat-sheet">
    <span class="cmd-key">SPEED</span>   : Low Latency<br>
    <span class="cmd-key">COST</span>    : Min Budget<br>
    <span class="cmd-key">STABLE</span>  : Max Redundancy<br>
    <span class="cmd-key">GROWTH</span>  : Balanced Expansion<br>
    <span class="cmd-key">FREEZE</span>  : Pause System<br>
    <span class="cmd-key">AUDIT</span>   : System Report<br>
    <span class="cmd-key">RESET</span>   : Resume Auto-Pilot
    </div>
    """, unsafe_allow_html=True)

decay = 0.90 + (0.09 * (1.0 - capex_pref))

# --- 7. INITIALIZE ---
if st.session_state.engine_v36 is None or st.session_state.engine_v36.num_agents != traffic_load:
    st.session_state.engine_v36 = BioEngine(300, 300, traffic_load)

engine = st.session_state.engine_v36
nodes_arr = np.array(st.session_state.nodes)

# RUN LOOP
for _ in range(12): 
    engine.step(st.session_state.nodes, latency_pref, decay, redundancy_pref, terrain_diff)

# --- 8. METRICS & ANALYSIS ---
if len(nodes_arr) > 1:
    mst_cost = minimum_spanning_tree(distance_matrix(nodes_arr, nodes_arr)).toarray().sum()
else:
    mst_cost = 0

cable_volume = np.sum(engine.trail_map > 1.0) / 10.0
capex_efficiency = min(100, (mst_cost / (cable_volume + 1)) * 100)

# --- 9. AGENT ACTION PHASE (POST-CALCULATION) ---
if is_agent:
    # 1. Propose NEW action based on CURRENT score
    proposed_params, is_waiting = st.session_state.agent_brain.propose_action(traffic_load, capex_efficiency)
    st.session_state.capex_key = proposed_params[0]
    st.session_state.redundancy_key = proposed_params[1]
    
    # 2. Learn from the result of the LAST action
    log_msg = st.session_state.agent_brain.learn(capex_efficiency, [capex_pref, redundancy_pref])
    
    if "Processing" in log_msg:
        pass 
    elif not st.session_state.agent_log or st.session_state.agent_log[-1] != f"Agent: {log_msg}":
        st.session_state.agent_log.append(f"Agent: {log_msg}")
        if len(st.session_state.agent_log) > 6: st.session_state.agent_log.pop(0)

# --- 10. DASHBOARD UI ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("### 🕸️ NET-OPT v36: SENTIENT")
    mode_label = "AUTONOMOUS" if is_agent else "MANUAL"
    st.caption(f"OPTIMIZATION TARGET: STEINER TREE APPROXIMATION | MODE: {mode_label}")

# METRICS (Now at top)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("DATA CENTERS", f"{len(nodes_arr)}")
m2.metric("MINIMUM VIABLE", f"{int(mst_cost)} km")
m3.metric("PROPOSED FIBER", f"{int(cable_volume)} km", delta=f"{int(cable_volume - mst_cost)}", delta_color="inverse")
m4.metric("PHYSICS C", f"{latency_pref}x")
m5.metric("EFFICIENCY", f"{int(capex_efficiency)}%")

# --- 11. VISUALIZATION TRIFECTA ---
col_vis1, col_vis2, col_stats = st.columns([1, 1, 1.2])

with col_vis1:
    st.markdown("**1. LATENCY TERRAIN (Solver)**")
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5)) 
    disp_map = np.log1p(engine.trail_map)
    ax1.imshow(disp_map, cmap='magma', origin='upper') 
    if len(nodes_arr) > 0:
        ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c='white', s=25, edgecolors='cyan', zorder=10)
    ax1.axis('off')
    fig1.tight_layout(pad=0)
    st.pyplot(fig1, width="stretch") 
    st.caption("Heatmap shows packet congestion. High Latency Priority forces straighter, hotter paths.")

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
    st.pyplot(fig2, width="stretch")
    st.caption("Green = Core Backbone. Blue = Failover. Terrain difficulty adds jitter to these paths.")

# 3. TELEMETRY STACK
with col_stats:
    st.markdown("**3. COST CONVERGENCE**")
    st.session_state.history.append({
        "MST Baseline": float(mst_cost), 
        "Bio-Solver": float(cable_volume),
        "Manual Budget": float(manual_budget)
    })
    if len(st.session_state.history) > 80: st.session_state.history.pop(0)
    
    chart_data = pd.DataFrame(st.session_state.history)
    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
    
    if not chart_data.empty:
        ax3.plot(chart_data["MST Baseline"], color='#444444', linestyle='--', linewidth=1, label="Optimal (MST)")
        ax3.plot(chart_data["Bio-Solver"], color='#00FF41', linewidth=1.5, label="Actual Cost")
        ax3.axhline(y=manual_budget, color='#FF4B4B', linestyle=':', linewidth=1.5, label=f"Target ({int(manual_budget)}k)")
        
    ax3.grid(color='#222', linestyle='-', linewidth=0.5)
    ax3.spines['bottom'].set_color('#444')
    ax3.spines['left'].set_color('#444')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(frameon=False, labelcolor='#888', fontsize=8, loc='upper right')
    
    st.pyplot(fig3, width="stretch")
    
    # 4. AGENT TERMINAL
    st.markdown("**4. AGENT TERMINAL**")
    status_dot = "🟢" if is_agent else "🔴"
    log_html = f"<div class='agent-terminal'>STATUS: {status_dot} LINK ESTABLISHED<br>"
    for line in st.session_state.agent_log:
        log_html += f"> {line}<br>"
    log_html += "<span style='animation: blink 1s infinite;'>_</span></div>"
    st.markdown(log_html, unsafe_allow_html=True)

# --- ANALYST SUMMARY (MOVED TO BOTTOM & COMPACT) ---
report_color = "#00FF41" if capex_efficiency > 50 else "#FFA500"
latency_status = "ULTRA-LOW (HFT)" if latency_pref > 3.0 else "STANDARD"
terrain_status = "HOSTILE" if terrain_diff > 0.3 else "STABLE"

st.markdown(f"""
<div style="border: 1px solid #333; padding: 5px; border-radius: 5px; background-color: #050505; margin-top: 10px; font-size: 11px;">
    <span style="color: #666; font-family: monospace;">[ANALYST SUMMARY]</span>
    <span style="color: {report_color}; font-family: monospace;">
    MODE: <b>{latency_status}</b> | TERRAIN: <b>{terrain_status}</b> | 
    EFFICIENCY: <b>{int(capex_efficiency)}%</b> | STATUS: <b>{ "OPTIMAL" if capex_efficiency > 60 else "ADJUSTING..."}</b>
    </span>
</div>
""", unsafe_allow_html=True)

# AUTO-LOOP
time.sleep(0.01)
st.rerun()
