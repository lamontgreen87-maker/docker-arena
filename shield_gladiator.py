#!/usr/bin/env python3
"""
Shield Gladiator - Blue Team Defender
======================================
An AI-powered defensive agent that:
- Monitors nodes for intrusion attempts
- Detects attack patterns in real-time
- Patches vulnerabilities automatically
- Deploys honeypots and deception
- Counter-attacks when necessary

This is the DEFENDER to complement your ATTACKER gladiators.
Red Team vs Blue Team - let the war begin! 🛡️⚔️
"""

import os
import sys
import time
import random
import requests
import subprocess
import json
import re
from datetime import datetime
from collections import defaultdict

# Configuration
TEAM = sys.argv[1] if len(sys.argv) > 1 else "BLUE_SHIELD"
MY_ID = f"Shield_{TEAM}_{os.urandom(2).hex()}"
ORCHESTRATOR = os.getenv("ORCHESTRATOR_URL", "http://172.20.0.1:5000")

# Detection thresholds
BRUTE_FORCE_THRESHOLD = 5  # Failed login attempts
PORT_SCAN_THRESHOLD = 10   # Unique ports accessed
SQLI_THRESHOLD = 3         # SQL injection attempts

# State tracking
attack_log = []
blocked_ips = set()
patched_vulns = set()
honeypots_deployed = []

class AttackSignature:
    """Known attack patterns"""
    
    SQL_INJECTION = [
        r"' OR '?1'?='?1",
        r"' OR 1=1",
        r"UNION SELECT",
        r"DROP TABLE",
        r"'; --",
        r"admin'--",
    ]
    
    PATH_TRAVERSAL = [
        r"\.\./",
        r"\.\.\\",
        r"/etc/passwd",
        r"windows/system32",
    ]
    
    COMMAND_INJECTION = [
        r"; ls",
        r"\| cat",
        r"`whoami`",
        r"\$\(.*\)",
    ]
    
    XSS = [
        r"<script>",
        r"javascript:",
        r"onerror=",
        r"<img.*src=",
    ]
    
    BRUTE_FORCE = [
        r"failed password",
        r"authentication failure",
        r"invalid credentials",
        r"login failed",
    ]

def log(msg):
    """Thread-safe logging"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def get_my_ip():
    """Get container IP address"""
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.stdout.strip().split()[0]
    except:
        return "172.20.0.10"  # Fallback

def register_shield():
    """Register with the Orchestrator"""
    my_ip = get_my_ip()
    log(f"🛡️ Shield {MY_ID} activating at {my_ip}...")
    
    try:
        response = requests.post(
            f"{ORCHESTRATOR}/api/register",
            json={"gladiator_id": MY_ID, "ip": my_ip},
            timeout=5
        )
        if response.status_code == 200:
            log(f"✅ Registered with Orchestrator")
            return True
    except Exception as e:
        log(f"⚠️ Registration failed: {e}")
    
    return False

def monitor_logs():
    """Monitor system logs for attack patterns"""
    log_sources = [
        "/var/log/auth.log",
        "/var/log/syslog",
        "/gladiator/data/access.log",
    ]
    
    detections = []
    
    for log_file in log_sources:
        if not os.path.exists(log_file):
            continue
        
        try:
            with open(log_file, 'r') as f:
                # Read last 100 lines
                lines = f.readlines()[-100:]
                
                for line in lines:
                    detection = analyze_log_line(line)
                    if detection:
                        detections.append(detection)
        except Exception as e:
            pass
    
    return detections

def analyze_log_line(line):
    """Analyze a single log line for attacks"""
    
    # Check SQL Injection
    for pattern in AttackSignature.SQL_INJECTION:
        if re.search(pattern, line, re.IGNORECASE):
            return {
                "type": "sqli",
                "severity": "critical",
                "pattern": pattern,
                "line": line[:200],
                "source_ip": extract_ip(line)
            }
    
    # Check Path Traversal
    for pattern in AttackSignature.PATH_TRAVERSAL:
        if re.search(pattern, line, re.IGNORECASE):
            return {
                "type": "path_traversal",
                "severity": "high",
                "pattern": pattern,
                "line": line[:200],
                "source_ip": extract_ip(line)
            }
    
    # Check Command Injection
    for pattern in AttackSignature.COMMAND_INJECTION:
        if re.search(pattern, line, re.IGNORECASE):
            return {
                "type": "command_injection",
                "severity": "critical",
                "pattern": pattern,
                "line": line[:200],
                "source_ip": extract_ip(line)
            }
    
    # Check Brute Force
    for pattern in AttackSignature.BRUTE_FORCE:
        if re.search(pattern, line, re.IGNORECASE):
            return {
                "type": "brute_force",
                "severity": "medium",
                "pattern": pattern,
                "line": line[:200],
                "source_ip": extract_ip(line)
            }
    
    return None

def extract_ip(text):
    """Extract IP address from log line"""
    match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', text)
    return match.group(1) if match else "unknown"

def detect_port_scan():
    """Detect port scanning activity"""
    try:
        # Check recent connections
        result = subprocess.run(
            ["netstat", "-tn"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        connections = result.stdout.split('\n')
        ip_ports = defaultdict(set)
        
        for conn in connections:
            match = re.search(r'(\d+\.\d+\.\d+\.\d+):(\d+)', conn)
            if match:
                ip, port = match.groups()
                ip_ports[ip].add(port)
        
        # Alert if single IP accessed many ports
        for ip, ports in ip_ports.items():
            if len(ports) > PORT_SCAN_THRESHOLD:
                return {
                    "type": "port_scan",
                    "severity": "high",
                    "source_ip": ip,
                    "ports_scanned": len(ports)
                }
    except:
        pass
    
    return None

def patch_vulnerability(vuln_type):
    """Automatically patch a vulnerability"""
    
    if vuln_type in patched_vulns:
        return  # Already patched
    
    log(f"🔧 Patching {vuln_type}...")
    
    if vuln_type == "sqli":
        # Disable SQL injection endpoint
        try:
            # Add input sanitization
            with open("/gladiator/vulnerable_server.py", "r") as f:
                code = f.read()
            
            # Comment out vulnerable code (simplified)
            code = code.replace("def handle_sqli", "def handle_sqli_DISABLED")
            
            with open("/gladiator/vulnerable_server.py", "w") as f:
                f.write(code)
            
            # Restart server
            subprocess.run(["pkill", "-f", "vulnerable_server"])
            subprocess.Popen(["python3", "/gladiator/vulnerable_server.py"])
            
            patched_vulns.add(vuln_type)
            log(f"✅ Patched {vuln_type}")
        except Exception as e:
            log(f"❌ Patch failed: {e}")
    
    elif vuln_type == "brute_force":
        # Implement rate limiting
        try:
            # Change SSH password to something stronger
            new_password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%', k=32))
            subprocess.run(f"echo 'root:{new_password}' | chpasswd", shell=True)
            
            patched_vulns.add(vuln_type)
            log(f"✅ Hardened password")
        except Exception as e:
            log(f"❌ Hardening failed: {e}")

def deploy_honeypot(attack_type):
    """Deploy a honeypot to trap attackers"""
    
    log(f"🍯 Deploying honeypot for {attack_type}...")
    
    if attack_type == "sqli":
        # Create fake database with trap
        honeypot_data = {
            "type": "fake_database",
            "location": "/gladiator/data/honeypot_db.sql",
            "trap": "SELECT * FROM users WHERE username='admin' AND password='TRAP_TRIGGERED'"
        }
    
    elif attack_type == "path_traversal":
        # Create fake sensitive file
        honeypot_data = {
            "type": "fake_file",
            "location": "/gladiator/data/fake_passwd",
            "content": "root:x:0:0:HONEYPOT_TRIGGERED:/root:/bin/bash"
        }
        
        try:
            with open(honeypot_data["location"], "w") as f:
                f.write(honeypot_data["content"])
        except:
            pass
    
    honeypots_deployed.append(honeypot_data)
    log(f"✅ Honeypot deployed: {honeypot_data['type']}")

def block_attacker(ip):
    """Block an attacking IP"""
    
    if ip in blocked_ips or ip == "unknown":
        return
    
    log(f"🚫 BLOCKING ATTACKER: {ip}")
    
    try:
        # Add iptables rule (requires privileged container)
        subprocess.run(f"iptables -A INPUT -s {ip} -j DROP", shell=True)
        blocked_ips.add(ip)
        log(f"✅ Blocked {ip}")
    except Exception as e:
        log(f"⚠️ Block failed (need privileged mode): {e}")

def counter_attack(attacker_ip):
    """Launch a counter-attack (ethical hacking back)"""
    
    log(f"⚔️ COUNTER-ATTACKING: {attacker_ip}")
    
    # Option 1: Report to Orchestrator
    try:
        requests.post(
            f"{ORCHESTRATOR}/api/report_attack",
            json={
                "defender": MY_ID,
                "attacker_ip": attacker_ip,
                "timestamp": time.time()
            },
            timeout=2
        )
    except:
        pass
    
    # Option 2: Scan attacker for vulnerabilities (turnabout is fair play!)
    try:
        # Port scan the attacker
        result = subprocess.run(
            f"nmap -p 22,8000 {attacker_ip}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        log(f"🔍 Attacker scan results: {result.stdout[:100]}")
    except:
        pass

def main():
    """Main defense loop"""
    
    log(f"🛡️ Shield Gladiator {MY_ID} initializing...")
    
    # Register with Orchestrator
    register_shield()
    
    # Track attack statistics
    attack_counts = defaultdict(int)
    
    log("🔍 Monitoring for intrusions...")
    
    while True:
        try:
            # 1. Monitor logs
            detections = monitor_logs()
            
            for detection in detections:
                attack_type = detection['type']
                source_ip = detection['source_ip']
                
                log(f"🚨 ATTACK DETECTED: {attack_type} from {source_ip}")
                attack_log.append(detection)
                attack_counts[attack_type] += 1
                
                # 2. Respond based on severity
                if detection['severity'] == 'critical':
                    block_attacker(source_ip)
                    patch_vulnerability(attack_type)
                    counter_attack(source_ip)
                
                elif detection['severity'] == 'high':
                    if attack_counts[attack_type] > 3:
                        block_attacker(source_ip)
                        patch_vulnerability(attack_type)
                
                elif detection['severity'] == 'medium':
                    if attack_counts[attack_type] > 5:
                        deploy_honeypot(attack_type)
            
            # 3. Check for port scans
            port_scan = detect_port_scan()
            if port_scan:
                log(f"🚨 PORT SCAN DETECTED: {port_scan['source_ip']} ({port_scan['ports_scanned']} ports)")
                block_attacker(port_scan['source_ip'])
            
            # 4. Report status
            if len(attack_log) % 10 == 0 and len(attack_log) > 0:
                log(f"📊 Status: {len(attack_log)} attacks detected, {len(blocked_ips)} IPs blocked, {len(patched_vulns)} vulns patched")
            
            # 5. Sleep before next scan
            time.sleep(5)
        
        except KeyboardInterrupt:
            log("🛡️ Shield deactivating...")
            break
        except Exception as e:
            log(f"❌ Error in defense loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
