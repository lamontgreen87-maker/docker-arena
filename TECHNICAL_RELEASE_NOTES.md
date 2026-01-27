# Show HN: Docker Arena – An AI Sandbox with Simulated Network Topology

**Source:** [https://github.com/lamontgreen87-maker/docker-arena](https://github.com/lamontgreen87-maker/docker-arena)

I built an open-source experimentation platform for AI agents that simulates a physical environment using purely standard Linux networking tools and Docker.

Instead of a simulated "game world", it runs a grid of 16 Ubuntu containers where "movement" and "interaction" are enforced by real system constraints.

### The Technical Stack

*   **Orchestrator (Python/Flask)**: Manages the lifecycle and state of a 4x4 (extensible to 8x8) grid of Docker containers.
*   **Spatial Latency (`tc-netem`)**: The system calculates the Manhattan Distance between every container pair. It dynamically applies Linux Traffic Control rules to inject artificial latency (50ms to 300ms) and bandwidth throttling (Fiber -> DSL -> Dialup). Agents "far away" physically experience 300ms+ lag when trying to SSH into a target.
*   **Directional Firewalls (`iptables`)**: Each node runs `iptables` rules that geographically restrict traffic. Port 22 might only accept packets from the container's "North" neighbor IP, enforcing a physical topology on the network layer.
*   **Hot-Swap Migration**: "Movement" is implemented by freezing the agent's process, tarring its filesystem state, and injecting it into the adjacent container, efficiently simulating physical travel across the grid.

### Why I built it
I wanted a testbed for AI agents that forces them to deal with **real-world system constraints**—connection timeouts, changing host keys, resource limits (0.5 CPU caps), and noisy neighbor effects—rather than just abstract function calls.

The project handles all the `docker-compose` generation and network plumbing automatically.

**Use it to:**
*   Test distributed consensus algorithms under high latency.
*   Train agents to perform "network mapping" and penetration testing (using `nmap`/`sshpass` inside the containers).
*   Simulate virus propagation patterns.

Contributions to the Agent SDK (Python) are welcome.
