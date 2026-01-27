import docker
import time
import os

client = docker.from_env()

def audit():
    print("--- Docker Audit ---")
    try:
        containers = client.containers.list(filters={"name": "arena_"})
        print(f"Found {len(containers)} arena containers.")
        for c in containers:
            print(f"[{c.name}] Status: {c.status}")
            
        # Check orchestrator logs
        orch = client.containers.get("arena_orchestrator")
        print("\n--- Orchestrator Logs (Last 10) ---")
        print(orch.logs(tail=10).decode())
        
        # Check arena_0_0 logs
        a00 = client.containers.get("arena_0_0")
        print("\n--- Arena_0_0 Logs (Last 5) ---")
        print(a00.logs(tail=5).decode())
        
        # Check if smart_gladiator is running in a00
        res = a00.exec_run("ps aux")
        print("\n--- Arena_0_0 Processes ---")
        print(res.output.decode())
        
    except Exception as e:
        print(f"Audit Error: {e}")

if __name__ == "__main__":
    audit()
