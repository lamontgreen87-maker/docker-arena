import docker
import random
import time

client = docker.from_env()

ALL_VULNS = [
    'RCE', 'LFI', 'SQLi', 'SSRF', 'XXE', 'DESERIAL', 
    'IDOR', 'AUTH_BYPASS', 'JWT', 'RACE', 'CORS', 
    'BUFFER', 'REDIRECT', 'ENV_LEAK',
    'SSTI', 'UPLOAD', 'NOSQLI', 'LOG_CRLF', 'MASS_ASSIGNMENT'
]

def randomize_vulnerability(container_name):
    try:
        container = client.containers.get(container_name)
        # 30% chance to reset, 70% to change
        if random.random() < 0.3:
            vulns = "NONE"
        else:
            count = random.randint(1, 4)
            vulns = ",".join(random.sample(ALL_VULNS, count))
        
        print(f"🎲 SCENARIO FUZZER: Randomizing {container_name} -> {vulns}")
        
        # We need to update the environment and RESTART the vulnerable server process
        # Since we can't easily change env vars on a running container without restart,
        # we will use a special trick: write a local config file that the server reads.
        
        container.exec_run(f"echo '{vulns}' > /gladiator/vulns.cfg")
        # The server needs to watch this file (if we want it truly live) 
        # but for now, we'll just restart the process.
        container.exec_run("pkill -f vulnerable_server.py")
        # The entrypoint.sh nohup will restart it? No, we started it manually in entrypoint.
        # Let's start it again.
        container.exec_run("nohup python3 vulnerable_server.py > /gladiator/http.log 2>&1 &", detach=True)
        
    except Exception as e:
        print(f"❌ Scenario Fuzzing Error for {container_name}: {e}")

if __name__ == "__main__":
    print("🎭 SCENARIO FUZZER: The Great Arena Randomizer is active.")
    while True:
        try:
            # Pick 2-3 random containers every 2 minutes
            containers = client.containers.list(filters={"name": "arena_node_"})
            if not containers:
                 containers = client.containers.list(filters={"name": "arena_"}) # Fallback
            
            # Filter specifically for the node containers (grid)
            nodes = [c.name for c in containers if "orchestrator" not in c.name and "gladiator" not in c.name]
            
            if nodes:
                targets = random.sample(nodes, min(len(nodes), 3))
                for target in targets:
                    randomize_vulnerability(target)
            
        except Exception as e:
            print(f"Loop Error: {e}")
            
        time.sleep(30) # 30 seconds (High-Speed Data Loop)
