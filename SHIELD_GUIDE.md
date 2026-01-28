# Shield Gladiator - Blue Team Defense Guide

## 🛡️ Overview
The Shield Gladiator is your **AI-powered defender** that protects nodes from Red Team attacks. It monitors in real-time, detects intrusions, patches vulnerabilities, and fights back!

## 🎯 What It Does

### **1. Intrusion Detection**
Monitors for attack patterns:
- **SQL Injection**: `' OR 1=1`, `UNION SELECT`
- **Path Traversal**: `../../etc/passwd`
- **Command Injection**: `; ls`, `| cat /etc/passwd`
- **Brute Force**: Failed login attempts
- **Port Scanning**: Unusual connection patterns

### **2. Automated Response**
When attacks are detected:
- **Critical threats**: Block IP + Patch vulnerability + Counter-attack
- **High threats**: Block IP after 3 attempts + Patch
- **Medium threats**: Deploy honeypot after 5 attempts

### **3. Vulnerability Patching**
Automatically fixes exploits:
- Disables vulnerable endpoints
- Hardens passwords
- Adds input sanitization
- Restarts services with patches

### **4. Honeypots**
Deploys traps for attackers:
- Fake databases with trigger alerts
- Fake sensitive files (`/etc/passwd`)
- Logs attacker behavior for analysis

### **5. Counter-Attacks**
Fights back ethically:
- Reports attacker to Orchestrator
- Scans attacker for vulnerabilities
- Gathers intelligence

## 🚀 Quick Start

### **Deploy a Shield**
```bash
# Start Shield on a specific node
docker exec -d arena_2_2 python3 /gladiator/shield_gladiator.py BLUE_SHIELD

# Check if it's running
docker exec arena_2_2 ps aux | grep shield
```

### **Monitor Shield Activity**
```bash
# Watch live logs
docker logs -f arena_2_2

# Expected output:
# [23:40:15] 🛡️ Shield Shield_BLUE_SHIELD_a3f2 activating at 172.20.2.12...
# [23:40:16] ✅ Registered with Orchestrator
# [23:40:16] 🔍 Monitoring for intrusions...
# [23:40:45] 🚨 ATTACK DETECTED: sqli from 172.20.1.10
# [23:40:45] 🚫 BLOCKING ATTACKER: 172.20.1.10
# [23:40:46] 🔧 Patching sqli...
# [23:40:47] ✅ Patched sqli
# [23:40:47] ⚔️ COUNTER-ATTACKING: 172.20.1.10
```

## 🎮 Red vs Blue Arena

### **Setup a Battle**
```bash
# Deploy Red Team (Attackers)
docker exec -d arena_0_0 python3 /gladiator/neural_gladiator.py RED

# Deploy Blue Team (Defenders)
docker exec -d arena_2_2 python3 /gladiator/shield_gladiator.py BLUE_SHIELD
docker exec -d arena_4_4 python3 /gladiator/shield_gladiator.py BLUE_SHIELD

# Watch the war unfold!
```

### **Scoring**
- **Red Team**: +10 points per node compromised
- **Blue Team**: +5 points per attack blocked
- **Red Team**: -5 points if detected and blocked
- **Blue Team**: -10 points if node is compromised

## 🔧 Configuration

### **Adjust Detection Thresholds**
Edit `shield_gladiator.py`:
```python
BRUTE_FORCE_THRESHOLD = 5   # Lower = more sensitive
PORT_SCAN_THRESHOLD = 10    # Lower = detect scans faster
SQLI_THRESHOLD = 3          # Lower = block SQLi sooner
```

### **Enable Privileged Mode (for IP blocking)**
Shields need elevated permissions to use `iptables`:
```yaml
# In docker-compose.yml
arena_2_2:
  privileged: true  # Allows iptables
  cap_add:
    - NET_ADMIN
```

### **Custom Attack Signatures**
Add your own patterns:
```python
class AttackSignature:
    CUSTOM_EXPLOIT = [
        r"your_pattern_here",
        r"another_pattern",
    ]
```

## 📊 Advanced Features

### **Machine Learning Detection**
Train the Shield to learn attack patterns:
```python
from sklearn.ensemble import IsolationForest

# Collect normal traffic
normal_traffic = []
for i in range(1000):
    normal_traffic.append(get_traffic_features())

# Train anomaly detector
detector = IsolationForest()
detector.fit(normal_traffic)

# Use in detection
if detector.predict([current_traffic]) == -1:
    log("🚨 ANOMALY DETECTED")
```

### **Adaptive Patching**
Shield learns which patches work:
```python
patch_success_rate = {}

def adaptive_patch(vuln_type):
    if patch_success_rate.get(vuln_type, 0) < 0.5:
        # Try alternative patch
        alternative_patch(vuln_type)
    else:
        # Use standard patch
        standard_patch(vuln_type)
```

### **Deception Tactics**
Advanced honeypots:
```python
def deploy_advanced_honeypot():
    # Fake admin panel
    create_fake_endpoint("/admin", "fake_login.html")
    
    # Fake database
    create_fake_db("users", fake_credentials)
    
    # Track who accesses it
    log_honeypot_access()
```

## 🎯 Real-World Applications

### **1. Security Testing**
Deploy Shields on your production Docker images:
```bash
# Test your app's defenses
docker run -d your_app:latest
docker exec -d your_app python3 shield_gladiator.py PROD_SHIELD

# Red Team attacks it
docker exec -d attacker neural_gladiator.py RED

# Shield protects and reports vulnerabilities
```

### **2. Training Simulation**
Train security teams:
- **Blue Team**: Deploy and configure Shields
- **Red Team**: Try to bypass defenses
- **Debrief**: Review attack logs and improve

### **3. Continuous Monitoring**
Run Shields 24/7 on staging environments:
```bash
# Deploy permanent Shield
docker-compose up -d shield_monitor

# Alerts sent to Slack/email when attacks detected
```

## 🚨 Safety Notes

**Shields are DEFENSIVE ONLY**:
- ✅ Monitor logs
- ✅ Block attackers
- ✅ Patch vulnerabilities
- ✅ Deploy honeypots
- ❌ Do NOT attack external systems
- ❌ Do NOT scan the internet

**Ethical Use**:
- Only deploy on YOUR OWN systems
- Get authorization before testing client systems
- Use counter-attacks for intelligence gathering only

## 🏆 Competition Ideas

### **Capture the Flag with Shields**
- Red Team: Capture flag at (5,5)
- Blue Team: Protect flag with Shields
- Winner: First to hold flag for 10 minutes

### **King of the Hill**
- Deploy Shields on central node (3,3)
- Red Teams attack from all sides
- Shield must survive 1 hour

### **Stealth Challenge**
- Red Team: Compromise nodes WITHOUT triggering Shields
- Blue Team: Detect ALL attacks
- Scoring: Stealth vs Detection accuracy

---

**Your arena is now a FULL RED vs BLUE BATTLEFIELD!** 🛡️⚔️🔴🔵

Deploy Shields, watch them defend, and let the AI war begin! 🤖🚨
