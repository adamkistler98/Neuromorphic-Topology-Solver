import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import random

# --- 1. GLOBAL SETTINGS ---
# Use standard dark background for plots to match Streamlit Dark Mode
plt.style.use('dark_background') 

st.set_page_config(
    page_title="NetOpt v38: Refined", 
    layout="wide", 
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# --- 2. MINIMAL CSS (Only for the Terminal & Cheat Sheet) ---
st.markdown("""
<style>
    /* AGENT TERMINAL - KEEPING THIS COOL */
    .agent-terminal {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #00FF41;
        background-color: #000;
        border: 1px solid #333;
        padding: 10px;
        height: 150px;
        overflow-y: auto;
        border-left: 3px solid #00FF41;
    }

    /* CHEAT SHEET - SMALL TEXT */
    .cmd-cheat-sheet {
        font-family: monospace;
        font-size: 10px;
        color: #888;
        margin-top: 10px;
    }
    .cmd-key { color: #00FF41; font-weight: bold; }
    
    /* HIDE DEPLOY BUTTON FOR CLEAN LOOK */
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# --- 3. BIO-ENGINE (PHYSICS KERNEL) ---
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

# --- 4. THE SENTIENT COMMAND AGENT ---
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

        # Precision Logic
        self.learning_rate = self.base_rate * 0.2 if current_score > 85 else self.base_rate
        
        # Stuck Detection (Quantum Jump)
        if self.fail_streak > 12:
            idx = random.choice([0, 1])
            self.current_params[idx] = random.uniform(0.2, 0.8)
            self.fail_streak = 0
            return self.current_params, False

        idx = random.choice([0, 1])
        change = random.choice([-self.learning_rate, self.learning_rate])

        # Enforce Mode Bias
        if self.override_mode == "SPEED" and idx == 1 and self.current_params[1] > 0.4: change = -abs(change)
        if self.override_mode == "COST" and idx == 0 and self.current_params[0] < 0.9: change = abs(change)
        if self.override_mode == "STABLE" and idx == 1 and self.current_params[1] < 1.2: change = abs(change)

        candidate = self.current_params.copy()
        candidate[idx] += change
        candidate[0] = np.clip(candidate[0], 0.01, 0.99)
        candidate[1] = np.clip(candidate[1], 0.1, 1.5)
        
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

# --- 5. STATE MANAGEMENT ---
if 'nodes' not in st.session_state: st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]]
if 'history' not in st.session_state: st.session_state.history = []
if 'agent_log' not in st.session_state: st.session_state.agent_log = ["System Initialized."]
if 'capex_key' not in st.session_state: st.session_state.capex_key = 0.8
if 'redundancy_key' not in st.session_state: st.session_state.redundancy_key = 0.7

# --- 6. SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🎛️ CONTROL PLANE")
control_mode = st.sidebar.radio("Operation Mode", ["🤖 Autonomous Agent", "Manual Operator"], horizontal=True)
is_agent = (control_mode == "🤖 Autonomous Agent")

st.sidebar.markdown("#### 1. SCENARIO CONFIG")
node_count = st.sidebar.slider("Network Scale", 3, 15, len(st.session_state.nodes))
preset = st.sidebar.selectbox("Load Preset", ["Diamond", "Grid", "Hub-Spoke", "Twin Cities", "Global Link", "Starlink", "Tri-State", "Pipeline", "The Void"])
manual_budget = st.sidebar.number_input("Target Budget (k)", value=250, step=10)

if st.sidebar.button("🎲 Randomize") or len(st.session_state.nodes) != node_count:
    st.session_state.nodes = [[random.randint(50, 250), random.randint(50, 250)] for _ in range(node_count)]
    st.session_state.history, st.session_state.engine_v38 = [], None
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    st.rerun()

if st.sidebar.button("⚠️ LOAD PRESET"):
    if preset == "Diamond": st.session_state.nodes = [[150, 50], [250, 150], [150, 250], [50, 150]]
    elif preset == "Grid": st.session_state.nodes = [[x, y] for x in range(80, 280, 70) for y in range(80, 280, 70)]
    elif preset == "Hub-Spoke": st.session_state.nodes = [[150, 150]] + [[150 + 110*np.cos(a), 150 + 110*np.sin(a)] for a in np.linspace(0, 2*np.pi, 7)[:-1]]
    elif preset == "Twin Cities": st.session_state.nodes = [[random.randint(50, 110), random.randint(50, 110)] for _ in range(4)] + [[random.randint(190, 250), random.randint(190, 250)] for _ in range(4)]
    elif preset == "Global Link": st.session_state.nodes = [[50, y] for y in np.linspace(50, 250, 5)] + [[250, y] for y in np.linspace(50, 250, 5)]
    elif preset == "Starlink": st.session_state.nodes = [[random.randint(20, 280), random.randint(20, 280)] for _ in range(12)]
    elif preset == "Tri-State": st.session_state.nodes = [[random.randint(40, 100), random.randint(40, 100)] for _ in range(3)] + [[random.randint(200, 260), random.randint(40, 100)] for _ in range(3)] + [[random.randint(120, 180), random.randint(200, 260)] for _ in range(3)]
    elif preset == "Pipeline": st.session_state.nodes = [[30 + i*20, 30 + i*20] for i in range(12)]
    elif preset == "The Void":
        nodes = []
        for i in range(10):
            nodes.append([random.randint(10, 290), random.randint(10, 40)])
            nodes.append([random.randint(10, 290), random.randint(260, 290)])
            nodes.append([random.randint(10, 40), random.randint(10, 290)])
            nodes.append([random.randint(260, 290), random.randint(10, 290)])
        st.session_state.nodes = random.sample(nodes, 12)
    
    st.session_state.history, st.session_state.engine_v38 = [], None
    if 'agent_brain' in st.session_state: del st.session_state.agent_brain
    st.rerun()

st.sidebar.markdown("---")
traffic_load = st.sidebar.slider("Traffic Load (Agents)", 1000, 8000, 5000)

if is_agent:
    if 'agent_brain' not in st.session_state: st.session_state.agent_brain = CommandAgent(st.session_state.capex_key, st.session_state.redundancy_key, len(st.session_state.nodes))
    st.sidebar.markdown("#### 2. AGENT SERVO")
    latency_pref, terrain_diff = 3.0, 0.1
    
    # BUFFERED FORM FOR CLI
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
    # Use 0 as placeholder efficiency, real calculation happens after loop
    proposed, _ = st.session_state.agent_brain.propose_action(traffic_load, 0)
    st.session_state.capex_key, st.session_state.redundancy_key = proposed[0], proposed[1]
else:
    st.session_state.capex_key, st.session_state.redundancy_key = capex_pref, redundancy_pref

# --- 7. CORE CALCULATION ---
if 'engine_v38' not in st.session_state or st.session_state.engine_v38 is None:
    st.session_state.engine_v38 = BioEngine(300, 300, traffic_load)

engine = st.session_state.engine_v38
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

# --- 8. UI RENDER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("### 🕸️ NET-OPT v38: REFINED")
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

# --- 9. ANALYST SUMMARY (Integrated Box) ---
report_color = "#00FF41" if capex_eff > 60 else "#FFA500"
st.markdown("---")
st.info(f"**ANALYST SUMMARY:** Network Operating at **{int(capex_eff)}% Efficiency**. Mode: **{latency_status if 'latency_status' in locals() else 'ADAPTIVE'}**. Recommended Action: **{'MAINTAIN' if capex_eff > 70 else 'OPTIMIZE'}**.")

time.sleep(0.01)
st.rerun()
