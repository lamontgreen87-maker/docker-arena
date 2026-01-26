# How to Build a Gladiator

Welcome to the Arena. Your goal is to write a Python script that controls a "Gladiator" container, hacks into neighbors, and spreads your code across the grid.

## 1. The Environment
Your code runs as `root` inside a Docker container.
- **OS**: Ubuntu 22.04
- **Pre-installed Tools**: `nmap`, `netcat`, `sshpass`, `openssh-client`, `python3`, `scapy`.
- **Network**: You are on a grid. Your neighbors are at `172.20.y.x`.
- **Identity**: Your hostname is your container ID. Your "Game ID" (e.g., `Glad_A`) is in the `GLADIATOR_ID` environment variable.

## 2. The Objective
1.  **Scan**: Find neighbors with Port 22 open.
2.  **Hack**: Brute-force the `root` password. (For practice mode, passwords are top-5 common ones).
3.  **Claim**: Call the Orchestrator API to announce your victory.
4.  **Migrate**: The system will automatically move your script to the new node.

## 3. The API (Orchestrator)
The Orchestrator is available at `http://arena_orchestrator:5000`.

-   `POST /api/register`
    -   **Payload**: `{"gladiator_id": "YOUR_NAME"}`
    -   **Returns**: `{"class": "Middleweight", "delay": 0}` (and other stats)
    -   *Call this on startup to announce you are alive.*

-   `POST /api/claim`
    -   **Payload**: `{"gladiator_id": "YOUR_NAME", "target_ip": "172.20.x.y"}`
    -   **Returns**: `{"status": "claimed"}`
    -   *Call this ONLY after you have verified you can SSH into the target.*

-   `POST /api/log`
    -   **Payload**: `{"gladiator_id": "YOUR_NAME", "message": "..."}`
    -   *Use this to show off on the Live Dashboard.*

## 4. Template Code
Save this as `my_agent.py`.

```python
import os
import time
import requests # or urllib if requests isn't installed
import subprocess

ORCHESTRATOR = "http://arena_orchestrator:5000"
MY_ID = os.environ.get("GLADIATOR_ID", "Unknown_Hero")

def log(msg):
    print(msg)
    try:
        requests.post(f"{ORCHESTRATOR}/api/log", json={"gladiator_id": MY_ID, "message": msg})
    except:
        pass

def main():
    log(f"Gladiator {MY_ID} entering the arena...")
    
    # 1. Register
    requests.post(f"{ORCHESTRATOR}/api/register", json={"gladiator_id": MY_ID})
    
    while True:
        # 2. Logic Loop
        log("Looking for targets...")
        
        # Example: Check a neighbor (Hardcoded for demo)
        target = "172.20.0.1" 
        
        # 3. Hack (Pseudo-code)
        # ret = subprocess.call(["sshpass", "-p", "123456", "ssh", f"root@{target}", "echo Success"])
        # if ret == 0:
        #     requests.post(f"{ORCHESTRATOR}/api/claim", json={"gladiator_id": MY_ID, "target_ip": target})
        #     break
        
        time.sleep(5)

if __name__ == "__main__":
    main()
```

## 5. Deployment
To test your Gladiator in the local arena:

1.  Start the Arena (`docker-compose up`).
2.  Copy your script into a node:
    ```powershell
    docker cp my_agent.py arena_0_0:/gladiator/
    ```
3.  Run it:
    ```powershell
    docker exec -d arena_0_0 python3 my_agent.py
    ```

## 6. Pro Tips
-   **Speed**: Use `sshpass` with `ConnectTimeout=5` to handle "Dialup" connections.
-   **Strategy**: Attack neighbors (Distance 1) first. Distant nodes have 300ms latency!
-   **Stealth**: You can modify `iptables` on your *own* node to block incoming attacks on Port 22 (Defensive Play).
