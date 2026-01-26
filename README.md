# Docker AI Arena 🏟️

**A "Capture the Flag" style competitive arena where AI agents live, fight, and migrate across a cluster of Docker containers.**

![Arena Banner](https://via.placeholder.com/800x200?text=Docker+AI+Arena)

## 📌 Overview
The **Docker Arena** simulates a distributed network of 16 nodes (4x4 grid).
You write a Python script (a "Gladiator") that spawns in one container.
Your goal:
1.  **Scan** your neighbors (`172.20.y.x`) for vulnerability.
2.  **Hack** into them by brute-forcing the root password (`ssh`).
3.  **Claim** the node via the Orchestrator API.
4.  **Migrate**: The system will literally *move* your code execution to the new node.

The Arena enforces:
-   **Network Latency**: Connections to distant nodes are throttled to 300ms (Dialup speeds).
-   **Resource Limits**: Gladiators are capped at 0.5 CPU / 256MB RAM.
-   **Rate Limiting**: SSH ports ban you if you spam hacking attempts too fast.

## 🚀 Quick Start
### Prerequisites
-   Docker Desktop installed and running.

### 1. Launch the Arena
```bash
docker-compose up -d --build
```
This will spin up:
-   16 "Node" Containers (Ubuntu 22.04)
-   1 Orchestrator (Flask API + Visualization)

### 2. View the Dashboard
Open [http://localhost:5000](http://localhost:5000) to see the live grid and logs.

### 3. Deploy a Gladiator
To verify the system works, deploy the included "Dummy Gladiator" (a smart bot that hunts neighbors):

```bash
# Inject the script into Node (0,0)
docker cp dummy_gladiator.py arena_0_0:/gladiator/

# Start the process
docker exec -d arena_0_0 python3 dummy_gladiator.py
```

### 4. Build Your Own
See [gladiator_creation_guide.md](gladiator_creation_guide.md) for the API reference and rules.

### 4. Victory Condition: The Golden Key 🏆
At the start of the match, one random node holds the **Golden Key**.
-   **Identify**: Hack nodes and look for `/gladiator/THE_KEY.txt`.
-   **Claim**: If found, you win!
-   **Specate**: The Dashboard highlights the Key's location with a yellow "KEY" indicator.

## 🛠️ Project Structure
-   `orchestrator/`: The Python Flask app that manages the game state and visualizer.
-   `node/`: The Dockerfile for the arena nodes (includes SSH, tools).
-   `directional_defense.py`: The firewall logic running on every node.
-   `dummy_gladiator.py`: A reference implementation of an AI agent.

## 📜 License
MIT License. Feel free to use this for hackathons, classes, or AI experiments!
