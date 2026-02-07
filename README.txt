# NET-Opt: Network Efficiency & Topology Optimizer
### v16 Strategic Planner | Steiner Tree Approximation Engine

![Python](https://img.shields.io/badge/Python-3.11%2B-00E5FF?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Streamlit-Enterprise-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Development-00E5FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-00E5FF?style=for-the-badge)

---

## 📉 Executive Overview

**NET-Opt** is a high-performance simulation engine designed to solve complex network topology problems where traditional algorithms fail. It does not just find the "shortest path" (Dijkstra); it finds the **Optimal Infrastructure Balance**—the "Sweet Spot" between **Capital Expenditure (CAPEX)**, **Redundancy**, and **Latency**.

By utilizing a **physarum-based transport solver**, NET-Opt autonomously generates **Steiner Tree approximations**. It visualizes how a network should be constructed to survive physical terrain constraints and signal interference while minimizing fiber-optic cabling costs.

> **Business Use Case:** A telecom architect needs to connect 5 regional data centers.
> * **Full Mesh?** Too expensive (High CAPEX).
> * **Single Line?** Too risky (Zero Redundancy).
> * **NET-Opt Solution:** Automatically generates a minimum-cost backbone with just enough loops to guarantee failover.

---

## 🛠️ Strategic Capabilities

### 1. Multi-Variable Optimization
NET-Opt solves for four competing constraints simultaneously:
* **CAPEX (Budget):** High decay rates prune inefficient paths, simulating strict fiber budgets.
* **Redundancy (Risk):** Wide sensor angles force the creation of "backup loops" for high-availability requirements.
* **Physics (Speed of Light):** Simulates signal propagation speed to prioritize low-latency (straight) routes for HFT scenarios.
* **Environment (Interference):** Injects "Terrain Noise" to simulate physical obstacles or EM interference, forcing robust routing.

### 2. The "Stealth" Command Console
A unified, zero-scroll dashboard designed for decision-makers:
* **Real-Time Telemetry:** Live comparison of "Optimal Baseline" (MST) vs. "Actual Proposed Cost."
* **Analyst Report:** Automated logic that grades your topology (e.g., *"Deployment Approved"* vs. *"Review Budget"*).
* **Dual-View Architect:**
    * **Left Screen:** *Latency Terrain* (Heatmap of bandwidth congestion).
    * **Right Screen:** *Blueprints* (Vectorized cable routes separating Backbone vs. Failover).

### 3. Enterprise Export
* **CSV Telemetry:** Export node coordinates and cost metrics for external GIS tools.
* **Snapshotting:** High-resolution capture of the finalized topology state.

---

## 🧠 The Engine: How It Works

NET-Opt rejects static pathfinding in favor of a **Stochastic Agent-Based Model (ABM)**:

| Business Logic | Simulation Variable | Biological Equivalent |
| :--- | :--- | :--- |
| **Budget / CAPEX** | **Decay Rate** | Path Evaporation |
| **Risk Tolerance** | **Sensor Angle** | Field of View |
| **Latency Priority** | **Agent Speed** | Metabolic Rate |
| **Terrain Difficulty** | **Noise Injection** | Random Jitter |

1.  **Exploration:** Thousands of "packet agents" flood the map, searching for data centers.
2.  **Reinforcement:** When a path connects two nodes effectively, it is reinforced (bandwidth increases).
3.  **Optimization:** The "Decay" factor continuously dissolves weak, unused paths, leaving behind only the most efficient trunk lines.

---

## 🚀 Installation & Deployment

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/net-opt.git](https://github.com/your-username/net-opt.git)
cd net-opt

pip install -r requirements.txt

streamlit run neuromorphic_topology_solver_final_v16.py

📦 NET-Opt
 ┣ 📜 neuromorphic_topology_solver_final_v16.py  # The Core Engine
 ┣ 📜 requirements.txt                           # Dependencies (numpy, scipy, matplotlib, streamlit)
 ┣ 📜 README.md                                  # Documentation
 ┗ 📂 /snapshots                                 # Exported topology maps

📜 License
Distributed under the MIT License. Engineered for educational and strategic planning purposes.

Engineered by Adam Kistler 
