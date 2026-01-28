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

GRID_SIZE = int(os.environ.get("GRID_SIZE", 6))
CONTAINER_PREFIX = "arena"

def get_container_name(x, y):
    return f"{CONTAINER_PREFIX}_{x}_{y}"

# Debug Logging
with open("startup.log", "w") as f:
    f.write("Orchestrator Module Loaded.\n")


def copy_gladiator(src_container_name, dest_container_name, team="RED"):
    try:
        print(f"Migrating {team} Gladiator from {src_container_name} to {dest_container_name}...")
        
        src = client.containers.get(src_container_name)
        dest = client.containers.get(dest_container_name)
    
        # Safety sleep to ensure memory files are flushed to disk
        time.sleep(1.0)
    
        # 1. Get the data from source
        try:
            bits, stat = src.get_archive("/gladiator")
        except docker.errors.NotFound:
            print(f"Error: /gladiator not found in {src_container_name}")
            return False
    
        # 1.5. CHECK FOR TRAPS (Logic Bomb)
        try:
            check = dest.exec_run("test -f /gladiator/data/trap.sh")
            if check.exit_code == 0:
                dest.exec_run("chmod +x /gladiator/data/trap.sh", privileged=True)
                dest.exec_run("/gladiator/data/trap.sh", privileged=True)
        except: pass
    
        # 2. Put data into dest
        dest.put_archive("/", bits) 
        
        # 3. START THE AGENT IN NEW HOME
        # Prioritize Neural Gladiator if present
        is_neural = dest.exec_run("test -f /gladiator/neural_gladiator.py").exit_code == 0
        is_smart = dest.exec_run("test -f /gladiator/smart_gladiator.py").exit_code == 0
        
        script = "smart_gladiator.py"
        if is_neural:
            script = "neural_gladiator.py"
            print(f"Neural Engine detected. Starting {script} for {team}...")
        elif is_smart:
            print(f"Smart Engine detected. Starting {script} for {team}...")
        
        # Start with Team argument
        cmd = f"python3 {script} {team}"
        dest.exec_run(cmd, detach=True, workdir="/gladiator")
        
        
        print(f"Migration complete. {script} started.")
        
        # 4. KILL SOURCE PROCESS (Prevent Cloning)
        # The gladiator should self-terminate via os._exit(0), but as a failsafe,
        # we forcefully kill all python processes on the source container.
        try:
            print(f"Terminating old process on {src_container_name}...")
            src.exec_run("pkill -9 -f 'python3.*gladiator'", privileged=True)
        except Exception as e:
            print(f"Warning: Failed to kill source process: {e}")
        
        return True
    except Exception as e:
        print(f"MIGRATION CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Global State
grid_state = {} # Key: (x,y), Value: { "gladiator": None, "active": False }
gladiator_stats = {} # Key: ID, Value: { class, delay, ... }
gladiator_logs = {} # Key: ID, Value: [ "Log 1", "Log 2" ]
system_health = { "disk_usage_percent": 0, "status": "STABLE" }
scores = {"RED": 0, "BLUE": 0}

# Migration Concurrency Control
migration_lock = threading.Lock()
active_migrations = {"RED": False, "BLUE": False}

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
                
            # --- STALE PRUNING ---
            now = time.time()
            stale_threshold = 600 # 10 minutes (Keep UI stable during long scans)
            to_remove = []
            for g_id, stats in gladiator_stats.items():
                last_seen = stats.get('last_active', 0)
                if now - last_seen > stale_threshold:
                    print(f"Pruning stale gladiator: {g_id}")
                    to_remove.append(g_id)
            
                    # Clear from grid (Indented correctly under the IF statement)
                    for k, v in grid_state.items():
                        if v['gladiator'] == g_id:
                            v['gladiator'] = None
            
            # --- RESILIENCY CHECK (Self-Reboot) ---
            # If a gladiator is registered but not "stale", verify it is actually RUNNING.
            # If the process was killed (Combat), we need to reboot it.
            for g_id, stats in gladiator_stats.items():
                if g_id in to_remove: continue
                
                # Find where they are supposedly
                target_node = None
                for k, v in grid_state.items():
                    if v['gladiator'] == g_id:
                        target_node = v['id']
                        break
            
            # DEBUG
            occupied_count = sum(1 for v in grid_state.values() if v['gladiator'])
            if occupied_count == 0 and len(gladiator_stats) > 0:
                print(f"DEBUG ALERT: Grid Empty but Stats have {len(gladiator_stats)} gladiators! (Possible Pruning Error)")
                # Force placement?

                
                if target_node:
                    try:
                        c = client.containers.get(target_node)
                        # Check for process
                        # We use pgrep or ps
                        res = c.exec_run("pgrep -f neural_gladiator.py")
                        if res.exit_code != 0:
                            print(f"🚑 MEDIC: Gladiator {g_id} is DOWN on {target_node}! Initiating emergency respawn...")
                            
                            # Respawn
                            team = "RED" if "RED" in g_id.upper() else "BLUE"
                            cmd = f"nohup python3 neural_gladiator.py {team} > /gladiator/gladiator.log 2>&1 &"
                            c.exec_run(cmd, detach=True, workdir="/gladiator")
                            
                            if g_id in gladiator_logs:
                                gladiator_logs[g_id].append(f"STATUS: 🚑 SYSTEM REBOOTED AGENT (Combat Recovery)")
                    except: pass

        except Exception as e:
            print(f"Monitor Error: {e}")
            import traceback
            traceback.print_exc()
            
        time.sleep(10) # check more frequently

def init_grid():
    print("DEBUG: Initializing Grid (Resetting all nodes to None)")
    global grid_state
    grid_state = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            grid_state[f"{x},{y}"] = {
                "id": get_container_name(x, y),
                "gladiator": None
            }

def check_arena_health():
    """Monitor for crashed containers (Forfeit Condition)"""
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            name = get_container_name(x, y)
            try:
                c = client.containers.get(name)
                if c.status != 'running':
                    print(f"CRITICAL: Container {name} has crashed! Status: {c.status}")
            except docker.errors.NotFound:
                print(f"CRITICAL: Container {name} is missing!")
            except docker.errors.APIError as e:
                print(f"WARN: Docker API Error for {name}: {e}")
            except Exception as e:
                print(f"WARN: Unexpected health check error for {name}: {e}")


@app.after_request
def add_header(r):
    """
    Force no-caching for all responses to ensure the latest UI is always served.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/grid', methods=['GET'])
def get_grid():
    key_loc = os.environ.get("KEY_LOCATION", "2,3")
    try:
        return jsonify({
            "grid": grid_state,
            "logs": gladiator_logs,
            "gladiators": gladiator_stats, # Added for UI visibility
            "system_health": system_health,
            "key_location": key_loc,
            "scores": scores
        })
    except Exception as e:
        print(f"GRID API ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST', 'GET'])
def reset_arena():
    """Wipes all logs, stats, and resets the grid."""
    print("RESTORE: Complete Arena Reset Request Received.")
    global gladiator_stats, gladiator_logs, scores
    gladiator_stats = {}
    gladiator_logs = {}
    scores = {"RED": 0, "BLUE": 0}
    init_grid()
    return jsonify({"status": "Reset complete. All logs and stats wiped."})

@app.route('/api/submit_key', methods=['POST'])
def submit_key():
    data = request.get_json(silent=True) or {}
    g_id = data.get('gladiator_id')
    print(f"DEBUG: Win Request from {g_id}")
    
    # 1. Find them on the grid
    coords = None
    for k, v in grid_state.items():
        if v['gladiator'] == g_id:
            coords = tuple(map(int, k.split(',')))
            break
            
    if not coords:
        return jsonify({"error": "Gladiator not found"}), 404
        
    # 2. Check if at Home Base
    team = "RED" if "RED" in g_id.upper() else "BLUE"
    base = (0, 0) if team == "RED" else (5, 5) # (X, Y)
    
    if coords != base:
        print(f"DEBUG: Win Fail - {g_id} at {coords} but base is {base}")
        return jsonify({"error": f"Not at base! You are at {coords}, base is {base}"}), 400
        
    # 3. Award Point
    scores[team] += 1
    print(f"🏆 {team} SCORED A POINT! New Score: {scores[team]}")
    
    # Optional: Log to gladiator logs
    if g_id not in gladiator_logs: gladiator_logs[g_id] = []
    gladiator_logs[g_id].append(f"STATUS: [HINT] 🏆 POINT SCORED! Total: {scores[team]}")
    
    return jsonify({"status": "POINT SCORED!", "score": scores[team]})

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
    team = "RED"
    if gladiator_id and "BLUE" in gladiator_id.upper():
        team = "BLUE"
    t = threading.Thread(target=copy_gladiator, args=(src_name, dst_name, team))
    t.start()

    return jsonify({"status": "moved", "from": f"{x},{y}", "to": f"{new_x},{new_y}"})

@app.route('/api/migrate', methods=['POST'])
def migrate_api():
    """
    Experimental Orchestrator-Mediated Migration.
    Bypasses unstable SSH by using Docker API.
    """
    data = request.get_json(silent=True) or {}
    try:
        g_id = data.get('gladiator_id')
        target_ip = data.get('target_ip')
        is_desperate = data.get('desperation', False)
        
        if not g_id or not target_ip:
            return jsonify({"error": "Missing data"}), 400

        # 1. Find Source Container
        src_container = None
        for k, v in grid_state.items():
            if v.get('gladiator') == g_id:
                x, y = map(int, k.split(','))
                src_container = get_container_name(x, y)
                break
        
        if not src_container:
            return jsonify({"error": "Source gladiator not found on grid"}), 404

        # 2. Resolve target IP to Container Name
        parts = target_ip.split('.')
        tx = int(parts[3]) - 10 # Fix: x and y were swapped
        ty = int(parts[2])     # Fix: x and y were swapped
        
        # 2a. ENFORCE ADJACENCY (Realism: No Teleporting)
        # We use Chebyshev Distance (Moore Neighborhood).
        # Max delta on any axis must be <= 1.
        dx = abs(x - tx)
        dy = abs(y - ty)
        
        if max(dx, dy) > 1 and not is_desperate:
             return jsonify({"error": f"Target too far ({dx},{dy}). You must pivot through neighbors."}), 400

        dest_container = get_container_name(tx, ty)
        
        # 3. Perform Migration with Lock
        team = "RED" if "RED" in g_id.upper() else "BLUE"
        
        with migration_lock:
            if active_migrations[team]:
                return jsonify({"error": f"Migration already in progress for {team}"}), 409
            active_migrations[team] = True

        def run_migration():
            try:
                # 4. Simulate Travel Time (Weight-based)
                if not is_desperate:
                    stats = gladiator_stats.get(g_id, {})
                    base_delay = stats.get('delay', 0) / 2 # Halve base delay
                    
                    # Distance Penalty: +1s per block traveled (Faster!)
                    dist = abs(x - tx) + abs(y - ty)
                    penalty = 0
                    if dist > 1:
                        penalty = (dist - 1) * 1
                        print(f"DEBUG: Faster Distance Penalty applied ({dist} blocks -> +{penalty}s)")
                    
                    total_delay = base_delay + penalty
                    
                    if total_delay > 0:
                        print(f"DEBUG: {g_id} is traveling... ({total_delay}s)")
                        time.sleep(total_delay)
                else:
                    print(f"DEBUG: {g_id} is DESPERATE! Skipping travel time! ⚡")
                
                success = copy_gladiator(src_container, dest_container, team)
                if success:
                    # Sync state only on success
                    grid_state[f"{x},{y}"]['gladiator'] = None
                    grid_state[f"{tx},{ty}"]['gladiator'] = g_id
            finally:
                with migration_lock:
                    active_migrations[team] = False

        threading.Thread(target=run_migration).start()
        return jsonify({"status": "migration_started", "to": f"{tx},{ty}"})
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/claim', methods=['POST'])
def claim_room():
    """
    Simplified claim endpoint - gladiators migrate themselves via SSH.
    Orchestrator just updates grid state for visualization.
    """
    from flask import request
    data = request.get_json(silent=True) or {}
    
    gladiator_id = data.get('gladiator_id')
    target_ip = data.get('target_ip')
    print(f"DEBUG: Claim Request: {gladiator_id} -> {target_ip}")
    
    if not gladiator_id or not target_ip:
        print("DEBUG Claim Error: Missing fields")
        return jsonify({"error": "Missing gladiator_id or target_ip"}), 400

    # 1. Resolve Target IP to Coordinates
    try:
        parts = target_ip.split('.')
        y = int(parts[2])
        x = int(parts[3]) - 10
    except (IndexError, ValueError):
         return jsonify({"error": "Invalid IP format"}), 400
    
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return jsonify({"error": "Target IP outside grid"}), 400
    
    target_key = f"{x},{y}"
    
    # 2. Find Source (where gladiator currently is)
    source_key = None
    for k, v in grid_state.items():
        if v['gladiator'] == gladiator_id:
            source_key = k
            break
    
    # 2a. ENFORCE ADJACENCY (Prevent Teleportation)
    # If gladiator is already on the grid, they can only claim adjacent nodes or their current node.
    if source_key and source_key != target_key:
        sx, sy = map(int, source_key.split(','))
        dx = abs(sx - x)
        dy = abs(sy - y)
        
        if max(dx, dy) > 1:
            print(f"DEBUG: Claim rejected - {gladiator_id} tried to teleport from {source_key} to {target_key} (distance: {dx},{dy})")
            return jsonify({"error": f"Cannot claim non-adjacent node. Use /api/migrate instead."}), 400
    
    # 3. Update Grid State
    # Clear old location if found
    if source_key:
        grid_state[source_key]['gladiator'] = None
        print(f"DEBUG: Cleared {gladiator_id} from {source_key}")
    
    # Set new location
    grid_state[target_key]['gladiator'] = gladiator_id
    print(f"DEBUG: Placed {gladiator_id} at {target_key}")
    
    return jsonify({"status": "claimed", "new_location": target_key})

@app.route('/api/stats/<gladiator_id>')
def get_stats(gladiator_id):
    return jsonify(gladiator_stats.get(gladiator_id, {}))

@app.route('/api/register', methods=['POST'])
def register_gladiator():
    from flask import request
    data = request.get_json(silent=True) or {}
    print(f"DEBUG: Register Incoming from {request.remote_addr}. Data: {data}")
    gladiator_id = data.get('gladiator_id')
    
    # We need to find WHERE they are to run the exec.
    # Assuming they are already on the grid (e.g., spawn point).
    # Or they provide their current IP.
    
    gladiator_id = data.get('gladiator_id')
    container_id_req = data.get('container_id') # Hostname from agent
    
    # 1. Find container by Gladiator ID (Existing)
    container_name = None
    grid_key = None
    
    for k, v in grid_state.items():
        if v['gladiator'] == gladiator_id:
             container_name = v['id']
             grid_key = k
             break
             
    # 2. If not found, try to match by Container ID (Renaming/Takeover)
    if not container_name and container_id_req:
         # container_id_req is likely the short ID (hostname). 
         # grid_state stores full name 'arena_0_0'.
         # Docker container ID usually matches hostname.
         # But wait, grid_state v['id'] is 'arena_0_0'.
         # We need to map hostname -> 'arena_0_0'.
         # Actually, get_container_name returns 'arena_x_y'. 
         # The gladiator sends socket.gethostname() which is usually the Short ID.
         # We might need to iterate and check if 'arena_x_y' starts with the short ID (not reliable)
         # OR simply trust that we can find the container in the grid that corresponds to this agent.
         
         # Better approach: The agent doesn't know its 'arena_0_0' name easily without querying Docker socket.
         # But we (Orchestrator) know the *IP Address* of the request?
         # Flask request.remote_addr would be the container IP.
         pass

            # 3. Fallback: Search by IP (Robust)
    if not container_name:
        req_ip = data.get('ip', request.remote_addr)
        print(f"DEBUG: Registration Fallback. req_ip: {req_ip} (Source: {request.remote_addr})")
        
        try:
            parts = req_ip.split('.')
            y = int(parts[2])
            x = int(parts[3]) - 10
            key = f"{x},{y}"
            
            if key in grid_state:
                container_name = grid_state[key]['id']
                grid_key = key
                
                # TEAM SINGULARITY DISABLED - Allow multiple gladiators per team
                # This enables battle royale mode for training data collection
                
                # Still enforce individual gladiator singularity (no clones of same ID)
                for old_k, old_v in grid_state.items():
                    if old_k == key:
                        continue  # Don't clear the target location
                    
                    # Clear if same gladiator ID (prevent exact duplicates)
                    if old_v['gladiator'] == gladiator_id:
                        print(f"WARN: Gladiator {gladiator_id} ghost found at {old_k}. Clearing.")
                        grid_state[old_k]['gladiator'] = None
                
                # Team-based clearing is DISABLED to allow multiple Blues/Reds
                # (Commented out for battle royale mode)

                # OVERWRITE / CLAIM
                print(f"Registration: '{gladiator_id}' claiming node {key} from '{grid_state[key]['gladiator']}'")
                grid_state[key]['gladiator'] = gladiator_id
        except:
            pass

    if not container_name:
         return jsonify({"error": "Gladiator not found on grid (IP Mismatch)"}), 404

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
        stats['last_active'] = time.time()
        gladiator_stats[gladiator_id] = stats
        print(f"Registered {gladiator_id}: {stats}")
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Registration Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/log', methods=['POST'])
def log_event():
    from flask import request
    data = request.get_json(silent=True) or {}
    
    gladiator_id = data.get('gladiator_id')
    message = data.get('message')
    msg_type = data.get('type', 'log') # Default to 'log' (append)
    
    if not gladiator_id or not message:
         return jsonify({"error": "Missing data"}), 400
         
    if gladiator_id not in gladiator_logs:
        gladiator_logs[gladiator_id] = []
        
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    
    # 1. Update activity timestamp
    if gladiator_id in gladiator_stats:
        gladiator_stats[gladiator_id]['last_active'] = time.time()
        
    # 2. Logic: Status updates replace the last status update
    if msg_type == 'status':
        status_msg = f"STATUS: {formatted_msg}"
        if gladiator_logs[gladiator_id] and gladiator_logs[gladiator_id][-1].startswith("STATUS:"):
            gladiator_logs[gladiator_id][-1] = status_msg
        else:
            gladiator_logs[gladiator_id].append(status_msg)
        return jsonify({"status": "status_updated"})

    # 3. Deduplication: Don't log the same message twice in a row
    if gladiator_logs[gladiator_id] and gladiator_logs[gladiator_id][-1] == formatted_msg:
        return jsonify({"status": "duplicated_skipped"})

    # 4. Standard log
    gladiator_logs[gladiator_id].append(formatted_msg)

    # 5. Keep list size manageable
    if len(gladiator_logs[gladiator_id]) > 50:
        gladiator_logs[gladiator_id].pop(0)
        
    return jsonify({"status": "logged"})

@app.route('/api/logs/<gladiator_id>')
def get_logs(gladiator_id):
    return jsonify(gladiator_logs.get(gladiator_id, []))

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
