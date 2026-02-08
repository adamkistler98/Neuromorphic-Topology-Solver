import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import random

# --- 1. GLOBAL SETTINGS & STYLING ---
plt.style.use('dark_background') 

st.set_page_config(
    page_title="NetOpt v37: Obsidian", 
    layout="wide", 
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# THE NUCLEAR CSS INJECTION
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #00FF41; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1A1A1A; }
    
    /* GLASSMORPHIC FOOTER */
    .status-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(5, 5, 5, 0.95);
        border-top: 1px solid #333;
        padding: 5px 20px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        z-index: 999;
        display: flex;
        justify-content: space-between;
        color: #888;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .heartbeat { color: #00FF41; animation: pulse 2s infinite; font-weight: bold; margin-right: 5px; }
    
    /* INPUT FIELDS & BUTTONS */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: #0A0A0A !important; color: #00FF41 !important; border: 1px solid #333 !important;
        font-family: 'Courier New', monospace;
    }
    div.stButton > button { background-color: #0A0A0A !important; color: #00FF41 !important; border: 1px solid #333 !important; width: 100%; }
    div.stButton > button:hover { border: 1px solid #00FF41 !important; background-color: #111 !important; }
    
    /* AGENT TERMINAL */
    .agent-terminal {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #00FF41;
        background-color: #000;
        border: 1px solid #333;
        padding: 8px;
        height: 140px;
        overflow-y: auto;
        margin-top: 10px;
        border-left: 2px solid #00FF41;
    }

    .cmd-cheat-sheet {
        font-family: 'Courier New', monospace;
        font-size: 10px;
        color: #555;
        margin-top: 10px;
        line-height: 1.5;
    }
    .cmd-key { color: #00FF41; }
    
    /* HIDE STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# --- 2. BIO-ENGINE (PHYSICS KERNEL) ---
class BioEngine:
    def __init__(self, width, height, num_agents):
        self.width, self.height = width, height
        self.num_agents = num_agents
        self.agents = np.zeros((num_agents, 3))
        self.agents[:, 0] = np.random.uniform(width*0.4, width*0.6, num_agents)
        self.agents[:, 1] = np.random.uniform(height*0.4, height*0.6, num_agents)
        self.agents[:, 2] = np.random.uniform(0, 2*np.pi, num_agents)
        self.trail_map = np.zeros((height, width))

    def step(self, nodes, speed, decay, sensor_angle, noise_level):
        sensor_dist = 9.0
        angles = self.agents[:, 2]
        def get_pos(a):
            x = (self.agents[:, 0] + np.cos(a) * sensor_dist) % self.width
            y = (self.agents[:, 1] + np.sin(a) * sensor_dist) % self.height
            return x.astype(int), y.astype(int)

        lx, ly = get_pos(angles - sensor_angle)
        cx, cy = get_pos(angles)
        rx, ry = get_pos(angles + sensor_angle)
        
        l_val, c_val, r_val = self.trail_map[ly, lx], self.trail_map[cy, cx], self.trail_map[ry, rx]
        jitter = np.random.uniform(-noise_level, noise_level, self.num_agents)
        
        move_fwd = (c_val > l_val) & (c_val > r_val)
        move_left = (l_val > c_val) & (l_val > r_val)
        move_right = (r_val > c_val) & (r_val > l_val)
        
        turn_rate = 0.5 / speed 
        self.agents[move_left, 2] -= turn_rate
        self.agents[move_right, 2] += turn_rate
        self.agents[~(move_fwd | move_left | move_right), 2] += jitter[~(move_fwd | move_left | move_right)]

        self.agents[:, 0] += np.cos(self.agents[:, 2]) * speed
        self.agents[:, 1] += np.sin(self.agents[:, 2]) * speed
        self.agents[:, 0] %= self.width
        self.agents[:, 1] %= self.height
        
        np.add.at(self.trail_map, (self.agents[:, 1].astype(int), self.agents[:, 0].astype(int)), 1.0) 
        
        for sx, sy in nodes:
            y_m, y_x = max(0, int(sy)-3), min(self.height, int(sy)+3)
            x_m, x_x = max(0, int(sx)-3), min(self.width, int(sx)+3)
            self.trail_map[y_m:y_x, x_m:x_x] += 5.0 

        self.trail_map = gaussian_filter(self.trail_map, sigma=0.6) * decay

# --- 3. THE SENTIENT COMMAND AGENT ---
class CommandAgent:
    def __init__(self, start_capex, start_redundancy, node_complexity=1):
        self.current_params = [start_capex, start_redundancy]
        self.best_params = [start_capex, start_redundancy]
        self.best_score = 0
        self.base_rate = 0.05 * (1 + (node_complexity * 0.1))
        self.learning_rate = self.base_rate
        self.cooldown = 0
        self.override_mode = None 
        self.is_frozen = False
        self.fail_streak = 0

    def process_command(self, cmd):
        cmd = cmd.lower()
        msg = "CMD: Unknown"
        if "freeze" in cmd: self.is_frozen = True; msg = "CMD: SYSTEM HALTED."
        elif "growth" in cmd: self.override_mode = "GROWTH"; self.current_params = [0.8, 0.8]; self.is_frozen = False; msg = "CMD: BALANCED GROWTH."
        elif "speed" in cmd: self.override_mode = "SPEED"; self.is_frozen = False; msg = "CMD: LOW LATENCY ENFORCED."
        elif "cost" in cmd: self.override_mode = "COST"; self.is_frozen = False; msg = "CMD: MINIMAL BUDGET ENFORCED."
        elif "stable" in cmd: self.override_mode = "STABLE"; self.is_frozen = False; msg = "CMD: MAX REDUNDANCY ENFORCED."
        elif "reset" in cmd: self.override_mode = None; self.is_frozen = False; msg = "CMD: AUTO-PILOT RESUMED."
        return msg, self.current_params

    def propose_action(self, current_load, current_score):
        if self.is_frozen or self.cooldown > 0:
            if self.cooldown > 0: self.cooldown -= 1
            return self.current_params, True

        # Precision Logic: Narrow search when efficiency is already high
        self.learning_rate = self.base_rate * 0.2 if current_score > 85 else self.base_rate
        
        # Stuck Detection: If baseline is stagnant, jump to find new territory
        if self.fail_streak > 12:
            idx = random.choice([0, 1])
            self.current_params[idx] = random.uniform(0.2, 0.8)
            self.fail_streak = 0
            return self.current_params, False

        idx = random.choice([0, 1])
        change = random.choice([-self.learning_rate, self.learning_rate])

        # Enforce Strategic Bias
        if self.override_mode == "SPEED" and idx == 1 and self.current_params[1] > 0.4: change = -abs(change)
        if self.override_mode == "COST" and idx == 0 and self.current_params[0] < 0.9: change = abs(change)
        if self.override_mode == "STABLE" and idx == 1 and self.current_params[1] < 1.2: change = abs(change)

        candidate = self.current_params.copy()
        candidate[idx] += change
        candidate[0] = np.clip(candidate[0], 0.01, 0.99)
        candidate[1] = np.clip(candidate[1], 0.1, 1.5)
        
        self.last_move = (idx, change)
        self.cooldown = 2
        return candidate, False

    def learn(self, score, params):
        if self.is_frozen: return "System Paused"
        delta = score - self.best_score
        if self.override_mode:
            self.best_score, self.best_params, self.current_params = score, params, params
            return f"EXECUTING {self.override_mode}"
        if delta > -3:
            if delta > 0: self.best_score = score; self.fail_streak = 0; return f"SUCCESS: New Baseline {int(score)}%"
            self.current_params = params; return "HOLD: Exploring..."
        else:
            self.fail_streak += 1; return "FAIL: Reverting..."

# --- 4. STATE MANAGEMENT ---
if 'nodes' not in st.session_state: st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]]
if 'history' not in st.session_state: st.session_state.history = []
if 'agent_log' not in st.session_state: st.session_state.agent_log = ["System Initialized."]
if 'capex_key' not in st.session_state: st.session_state.capex_key = 0.8
if 'redundancy_key' not in st.session_state: st.session_state.redundancy_key = 0.7

# --- 5. SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🎛️ CONTROL PLANE")
control_mode = st.sidebar.radio("Operation Mode", ["🤖 Autonomous Agent", "Manual Operator"], horizontal=True)
is_agent = (control_mode == "🤖 Autonomous Agent")

st.sidebar.markdown("#### 1. SCENARIO CONFIG")
node_count = st.sidebar.slider("Network Scale", 3, 15, len(st.session_state.nodes))
preset = st.sidebar.selectbox("Load Preset", ["Diamond", "Grid", "Hub-Spoke", "Twin Cities", "Global Link", "Starlink", "Tri-State", "Pipeline", "The Void"])
manual_budget = st.sidebar.number_input("Target Budget (k)", value=250, step=10)

if st.sidebar.button("🎲 Randomize") or len(st.session_state.nodes) != node_count:
    st.session_state.nodes = [[random.randint(50, 250), random.randint(50, 250)] for _ in range(node_count)]
    st.session_state.history, st.session_state.engine_v37 = [], None
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    st.rerun()

if st.sidebar.button("⚠️ LOAD PRESET"):
    if preset == "Twin Cities": st.session_state.nodes = [[random.randint(50, 110), random.randint(50, 110)] for _ in range(4)] + [[random.randint(190, 250), random.randint(190, 250)] for _ in range(4)]
    elif preset == "Global Link": st.session_state.nodes = [[50, y] for y in np.linspace(50, 250, 5)] + [[250, y] for y in np.linspace(50, 250, 5)]
    elif preset == "Starlink": st.session_state.nodes = [[random.randint(20, 280), random.randint(20, 280)] for _ in range(12)]
    elif preset == "Pipeline": st.session_state.nodes = [[30 + i*20, 30 + i*20] for i in range(12)]
    st.session_state.history, st.session_state.engine_v37 = [], None
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    st.rerun()

st.sidebar.markdown("---")
traffic_load = st.sidebar.slider("Traffic Load (Agents)", 1000, 8000, 5000)

if is_agent:
    if 'agent_brain' not in st.session_state: st.session_state.agent_brain = CommandAgent(st.session_state.capex_key, st.session_state.redundancy_key, len(st.session_state.nodes))
    st.sidebar.markdown("#### 2. AGENT SERVO")
    latency_pref, terrain_diff = 3.0, 0.1
    with st.sidebar.form(key='cli_form', clear_on_submit=True):
        user_cmd = st.text_input("CONSOLE >_", placeholder="Enter Command...")
        if st.form_submit_button("EXECUTE"):
            msg, params = st.session_state.agent_brain.process_command(user_cmd)
            st.session_state.agent_log.append(msg); st.rerun()
    st.sidebar.markdown('<div class="cmd-cheat-sheet"><span class="cmd-key">SPEED</span> | <span class="cmd-key">COST</span> | <span class="cmd-key">STABLE</span> | <span class="cmd-key">FREEZE</span> | <span class="cmd-key">RESET</span></div>', unsafe_allow_html=True)
else:
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    st.sidebar.markdown("#### 2. MANUAL OVERRIDE")
    latency_pref = st.sidebar.slider("Signal Speed (C)", 1.0, 5.0, 2.0)
    terrain_diff = st.sidebar.slider("Environment Noise", 0.05, 0.5, 0.1)

capex_pref = st.sidebar.slider("CAPEX Limit (Decay)", 0.0, 1.0, value=float(np.clip(st.session_state.capex_key, 0.0, 1.0)), key="c_sld")
redundancy_pref = st.sidebar.slider("Risk Tolerance (Angle)", 0.1, 1.5, value=float(np.clip(st.session_state.redundancy_key, 0.1, 1.5)), key="r_sld")

if is_agent:
    proposed, _ = st.session_state.agent_brain.propose_action(traffic_load, 0)
    st.session_state.capex_key, st.session_state.redundancy_key = proposed[0], proposed[1]
else:
    st.session_state.capex_key, st.session_state.redundancy_key = capex_pref, redundancy_pref

# --- 6. CORE CALCULATION ---
if 'engine_v37' not in st.session_state or st.session_state.engine_v37 is None:
    st.session_state.engine_v37 = BioEngine(300, 300, traffic_load)

engine = st.session_state.engine_v37
for _ in range(12): engine.step(st.session_state.nodes, latency_pref, 0.9 + (0.09*(1-capex_pref)), redundancy_pref, terrain_diff)

nodes_arr = np.array(st.session_state.nodes)
mst_cost = minimum_spanning_tree(distance_matrix(nodes_arr, nodes_arr)).toarray().sum() if len(nodes_arr)>1 else 0
cable_volume = np.sum(engine.trail_map > 1.0) / 10.0
capex_eff = min(100, (mst_cost / (cable_volume + 1)) * 100)

if is_agent:
    log_msg = st.session_state.agent_brain.learn(capex_eff, [capex_pref, redundancy_pref])
    if "Process" not in log_msg:
        if not st.session_state.agent_log or st.session_state.agent_log[-1] != f"Agent: {log_msg}":
            st.session_state.agent_log.append(f"Agent: {log_msg}")

# --- 7. UI RENDER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("### 🕸️ NET-OPT v37: OBSIDIAN")
    st.caption(f"STATUS: ACTIVE | COMPLEXITY: {len(nodes_arr)} NODES | MODE: {control_mode.upper()}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("NODES", len(nodes_arr)); m2.metric("MINIMUM", f"{int(mst_cost)}k")
m3.metric("CURRENT", f"{int(cable_volume)}k", delta=int(cable_volume-mst_cost), delta_color="inverse")
m4.metric("PHYSICS", f"{latency_pref}x"); m5.metric("EFFICIENCY", f"{int(capex_eff)}%")

v1, v2, v3 = st.columns([1, 1, 1.2])
with v1:
    st.markdown("**1. LATENCY TERRAIN**")
    f1, a1 = plt.subplots(figsize=(3,3))
    a1.imshow(np.log1p(engine.trail_map), cmap='magma'); a1.scatter(nodes_arr[:,0], nodes_arr[:,1], c='white', s=20); a1.axis('off')
    st.pyplot(f1, width="stretch")

with v2:
    st.markdown("**2. BLUEPRINT**")
    f2, a2 = plt.subplots(figsize=(3,3))
    a2.set_xlim(0,300); a2.set_ylim(300,0)
    yc, xc = np.where(engine.trail_map >= 3.0)
    a2.scatter(xc, yc, c='#00FF41', s=0.5, alpha=0.8)
    a2.scatter(nodes_arr[:,0], nodes_arr[:,1], c='#00FF41', marker='s', s=40); a2.axis('off')
    st.pyplot(f2, width="stretch")

with v3:
    st.markdown("**3. CONVERGENCE**")
    st.session_state.history.append({"Min": float(mst_cost), "Actual": float(cable_volume)})
    if len(st.session_state.history)>60: st.session_state.history.pop(0)
    df = pd.DataFrame(st.session_state.history)
    f3, a3 = plt.subplots(figsize=(4, 2.2))
    if not df.empty:
        a3.plot(df["Min"], color='#444', linestyle='--', label="Ideal")
        a3.plot(df["Actual"], color='#00FF41', label="Spend")
        a3.axhline(y=manual_budget, color='#FF4B4B', linestyle=':', label="Target")
    a3.legend(fontsize=7, loc='upper right'); a3.axis('off')
    st.pyplot(f3, width="stretch")
    
    st.markdown("**4. AGENT TERMINAL**")
    log_html = f"<div class='agent-terminal'>STATUS: {'🟢' if is_agent else '🔴'} LINK ONLINE<br>"
    for line in st.session_state.agent_log[-5:]: log_html += f"> {line}<br>"
    st.markdown(log_html + "</div>", unsafe_allow_html=True)

# THE SENTIENT FOOTER
st.markdown(f"""
<div class="status-footer">
    <div><span class="heartbeat">●</span> MODE: {preset.upper()} | LATENCY: { "ULTRA-LOW" if latency_pref>3 else "STANDARD" } | EFFICIENCY: {int(capex_eff)}%</div>
    <div>SENTIENT_OPTIMIZER: {"ACTIVE" if is_agent else "STANDBY"}</div>
</div>
""", unsafe_allow_html=True)

time.sleep(0.01)
st.rerun()
