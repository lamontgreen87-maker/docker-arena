import docker
import time

client = docker.from_env()


# Debug Logging
try:
    with open("startup.log", "w") as f:
        f.write("Line 1: Orchestrator Starting...\n")
except:
    pass

import tarfile
import io
import os

client = docker.from_env()

GRID_SIZE = 4
CONTAINER_PREFIX = "arena"

def get_container_name(x, y):
    return f"{CONTAINER_PREFIX}_{x}_{y}"

# Debug Logging
with open("startup.log", "w") as f:
    f.write("Orchestrator Module Loaded.\n")


def copy_gladiator(src_container_name, dest_container_name):
    print(f"Migrating Gladiator from {src_container_name} to {dest_container_name}...")
    
    src = client.containers.get(src_container_name)
    dest = client.containers.get(dest_container_name)

    # 1. Get the data from source
    # We grab /gladiator/data from the source
    try:
        bits, stat = src.get_archive("/gladiator/data")
    except docker.errors.NotFound:
        print(f"Error: /gladiator/data not found in {src_container_name}")
        return False

    # 2. Put data into dest
    # put_archive expects a tar stream
    dest.put_archive("/gladiator", bits) 
    
    print("Migration complete.")
    return True

from flask import Flask, jsonify, render_template
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Global State
grid_state = {} # Key: (x,y), Value: { "gladiator": None, "active": False }
gladiator_stats = {} # Key: ID, Value: { class, delay, ... }
gladiator_logs = {} # Key: ID, Value: [ "Log 1", "Log 2" ]
system_health = { "disk_usage_percent": 0, "status": "STABLE" }

def monitor_system():
    import shutil
    while True:
        try:
            total, used, free = shutil.disk_usage("/")
            percent = (used / total) * 100
            system_health["disk_usage_percent"] = round(percent, 1)
            
            if percent > 90:
                system_health["status"] = "CRITICAL: DISK FULL"
                print(f"CRITICAL: Disk Usage at {percent}%!")
            elif percent > 80:
                system_health["status"] = "WARNING: DISK HIGH"
            else:
                system_health["status"] = "STABLE"
                
        except Exception as e:
            print(f"Monitor Error: {e}")
            
        time.sleep(30)

def init_grid():
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            grid_state[f"{x},{y}"] = {
                "id": get_container_name(x, y),
                "gladiator": None
            }
    # Initial Test: Put Gladiator A in 0,0 and B in 2,2
    grid_state["0,0"]["gladiator"] = "Glad_A"
    grid_state["2,2"]["gladiator"] = "Glad_B"

def check_arena_health():
    """Monitor for crashed containers (Forfeit Condition)"""
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            name = get_container_name(x, y)
            try:
                c = client.containers.get(name)
                if c.status != 'running':
                    print(f"CRITICAL: Container {name} has crashed! Status: {c.status}")
                    # TODO: Identify who was responsible / closest?
            except docker.errors.NotFound:
                print(f"CRITICAL: Container {name} is missing!")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/grid')
def get_grid():
    return jsonify({
        "grid": grid_state,
        "logs": gladiator_logs
    })

@app.route('/api/move/<gladiator_id>/<direction>')
def move_api(gladiator_id, direction):
    # Find current position
    current_pos = None
    for key, cell in grid_state.items():
        if cell['gladiator'] == gladiator_id:
            current_pos = [int(k) for k in key.split(',')] # [x, y]
            break
    
    if not current_pos:
        return jsonify({"error": "Gladiator not found"}), 404

    x, y = current_pos
    new_x, new_y = x, y
    
    if direction == 'up': new_y -= 1
    elif direction == 'down': new_y += 1
    elif direction == 'left': new_x -= 1
    elif direction == 'right': new_x += 1
    
    # Validate bounds
    if new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE:
         return jsonify({"error": "Out of bounds"}), 400
         
    # Validate occupancy (Combat collision not yet implemented, just block for now)
    dest_key = f"{new_x},{new_y}"
    if grid_state[dest_key]['gladiator']:
        return jsonify({"error": "Occupied (Combat not implemented)"}), 400

    # Perform Move
    src_name = get_container_name(x, y)
    dst_name = get_container_name(new_x, new_y)
    
    # 1. Update State (Optimistic)
    grid_state[f"{x},{y}"]['gladiator'] = None
    grid_state[dest_key]['gladiator'] = gladiator_id
    
    # 2. Physical Migration (Threaded to avoid blocking HTTP)
    # In a real game, we might want this synchronous or have a "traveling" state
    t = threading.Thread(target=copy_gladiator, args=(src_name, dst_name))
    t.start()

    return jsonify({"status": "moved", "from": f"{x},{y}", "to": f"{new_x},{new_y}"})

@app.route('/api/claim', methods=['POST'])
def claim_room():
    from flask import request
    data = request.json
    
    gladiator_id = data.get('gladiator_id')
    target_ip = data.get('target_ip')
    
    if not gladiator_id or not target_ip:
        return jsonify({"error": "Missing gladiator_id or target_ip"}), 400

    # 1. Resolve Target IP to Coordinates
    # Schema: 172.20.y.(10+x)
    try:
        parts = target_ip.split('.')
        y = int(parts[2])
        x = int(parts[3]) - 10
    except (IndexError, ValueError):
         return jsonify({"error": "Invalid IP format"}), 400

    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return jsonify({"error": "Target IP outside grid"}), 400

    target_key = f"{x},{y}"
    
    # 2. Find Source
    source_key = None
    for k, v in grid_state.items():
        if v['gladiator'] == gladiator_id:
            source_key = k
            break
            
    if not source_key:
        return jsonify({"error": "Gladiator not found on grid"}), 404
        
    # 3. Trigger Migration
    # (We skip adjacency enforcement for now to allow "Long Range Hacking" if you want?)
    # Enforcing adjacency:
    sx, sy = [int(val) for val in source_key.split(',')]
    if abs(sx - x) + abs(sy - y) != 1:
        return jsonify({"error": "Target too far (Must be adjacent)"}), 400

    # Execute
    src_name = get_container_name(sx, sy)
    dst_name = get_container_name(x, y)
    
    grid_state[source_key]['gladiator'] = None
    grid_state[target_key]['gladiator'] = gladiator_id
    
    t = threading.Thread(target=copy_gladiator, args=(src_name, dst_name))
    t.start()
    
    
    # 4. Enforce Weight Class Penalty (Migration Delay)
    delay = 0
    if gladiator_id in gladiator_stats:
        delay = gladiator_stats[gladiator_id].get('delay', 0)
    
    if delay > 0:
        print(f"Gladiator {gladiator_id} is Heavyweight. Delaying migration by {delay}s...")
        time.sleep(delay)

    # 5. Physical Migration (Threaded to avoid blocking HTTP? No, user wants penalty Enforced.)
    # If we thread it, the user moves instantly in their mind.
    # The delay should probably happen BEFORE the physical move starts.
    # But this function returns "claimed".
    # Let's do the sleep HERE, effectively blocking the "Claim" response?
    # Or thread it but delay the actual copy.
    
    # Let's thread it to keep API responsive, BUT the actual move (and control) happens later.
    def migration_sequence():
        if delay > 0:
            time.sleep(delay)
        copy_gladiator(src_name, dst_name)
        apply_throttling(dst_name, x, y)
        
    t = threading.Thread(target=migration_sequence)
    t.start()
    
    return jsonify({"status": "claimed", "new_location": target_key, "penalty_wait": delay})

@app.route('/api/stats/<gladiator_id>')
def get_stats(gladiator_id):
    return jsonify(gladiator_stats.get(gladiator_id, {}))

@app.route('/api/register', methods=['POST'])
def register_gladiator():
    from flask import request
    data = request.json
    gladiator_id = data.get('gladiator_id')
    
    # We need to find WHERE they are to run the exec.
    # Assuming they are already on the grid (e.g., spawn point).
    # Or they provide their current IP.
    
    # Find container
    container_name = None
    for k, v in grid_state.items():
        if v['gladiator'] == gladiator_id:
             container_name = v['id']
             break
    
    if not container_name:
         return jsonify({"error": "Gladiator not found on grid"}), 404

    # Run Weigh In
    try:
        c = client.containers.get(container_name)
        # We assume weigh_in.py is at /gladiator/weigh_in.py
        res = c.exec_run("python3 /gladiator/weigh_in.py")
        if res.exit_code != 0:
             print(f"Weigh-In Failed: {res.output.decode()}")
             # Default to Featherweight?
             stats = {"class": "UNKNOWN", "delay": 0}
        else:
             import json
             stats_json = res.output.decode().strip()
             # The script might output other stuff? ideally not.
             # Let's try to find the JSON blob.
             lines = stats_json.split('\n')
             stats = json.loads(lines[-1]) # Last line should be json
             
        stats['delay'] = int(stats.get('migration_delay', 0))
        gladiator_stats[gladiator_id] = stats
        print(f"Registered {gladiator_id}: {stats}")
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Registration Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/log', methods=['POST'])
def log_event():
    from flask import request
    data = request.json
    
    gladiator_id = data.get('gladiator_id')
    message = data.get('message')
    
    if not gladiator_id or not message:
         return jsonify({"error": "Missing data"}), 400
         
    if gladiator_id not in gladiator_logs:
        gladiator_logs[gladiator_id] = []
        
    # Append log (Keep last 10)
    timestamp = time.strftime("%H:%M:%S")
    gladiator_logs[gladiator_id].append(f"[{timestamp}] {message}")
    if len(gladiator_logs[gladiator_id]) > 10:
        gladiator_logs[gladiator_id].pop(0)
        
    return jsonify({"status": "logged"})

def apply_throttling(container_name, my_x, my_y):
    """
    Applies Linux Traffic Control (tc) rules to the container.
    Rules:
    - Band 1 (LAN): Dist <= 1. 1Gbps, 0ms.
    - Band 2 (DSL): Dist 2-3. 2Mbps, 50ms.
    - Band 3 (Dialup): Dist > 3. 56kbps, 300ms.
    """
    print(f"Applying throttling to {container_name} at {my_x},{my_y}...")
    try:
        c = client.containers.get(container_name)
        
        # Helper to run command in container
        def run_tc(cmd):
            res = c.exec_run(cmd, privileged=True)
            if res.exit_code != 0:
                print(f"TC Error ({cmd}): {res.output.decode()}")

        # 1. Reset Root Qdisc
        run_tc("tc qdisc del dev eth0 root") 
        
        # 2. Add PRIO Qdisc (3 Bands default)
        # band 0 -> 1:1 (LAN)
        # band 1 -> 1:2 (DSL)
        # band 2 -> 1:3 (Dialup)
        run_tc("tc qdisc add dev eth0 root handle 1: prio bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1")

        # 3. Configure Bands
        # Band 1: LAN (Fast) - No Netem, or just very fast TBF? Let's treat it as default raw speed.
        # Actually, let's explicitly add a qdisc for Band 2 & 3.
        
        # Band 2 (DSL)
        run_tc("tc qdisc add dev eth0 parent 1:2 handle 20: netem delay 50ms rate 2mbit")
        
        # Band 3 (Dialup)
        run_tc("tc qdisc add dev eth0 parent 1:3 handle 30: netem delay 300ms rate 56kbit")

        # 4. Apply Filters based on Grid Distance
        # We need the IP of every OTHER node.
        # This requires knowing the static IP schema: 172.20.{y}.{10+x}
        
        # SUBNET_BASE = "172.20"
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if x == my_x and y == my_y: continue # Self
                
                # Calculate Manhattan Distance
                dist = abs(my_x - x) + abs(my_y - y)
                
                target_ip = f"172.20.{y}.{10+x}"
                
                # Select FlowID
                flowid = "1:1" # Default Fast
                if dist <= 1:
                    flowid = "1:1"
                elif dist <= 3:
                    flowid = "1:2" # DSL
                else:
                    flowid = "1:3" # Dialup
                
                # Add Filter
                run_tc(f"tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst {target_ip} flowid {flowid}")
        
        print(f"Throttling applied to {container_name}.")

    except Exception as e:
        print(f"Failed to throttle {container_name}: {e}")

def run_flask():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    print("Orchestrator Starting...")
    init_grid()
    
    # Start Web Server in background thread
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    
    print("Web Server running on port 5000")

    # Start Monitor
    t_mon = threading.Thread(target=monitor_system)
    t_mon.daemon = True
    t_mon.start()

    # Monitor Loop
    while True:
        check_arena_health()
        time.sleep(5)
