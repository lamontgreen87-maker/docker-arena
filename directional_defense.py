import os
import subprocess
import socket
import threading

# Config
GRID_SIZE = 8
BASE_PORT = 9000
# Directions: N, NE, E, SE, S, SW, W, NW
# Offsets (dx, dy) where x is col, y is row.
# Coordinate system: Top-Left is 0,0. y increases Down. x increases Right.
# North Neighbor is (x, y-1). 
DIRECTIONS = [
    (0, -1), # North
    (1, -1), # North-East
    (1, 0),  # East
    (1, 1),  # South-East
    (0, 1),  # South
    (-1, 1), # South-West
    (-1, 0), # West
    (-1, -1) # North-West
]

def get_my_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def ip_to_coords(ip):
    # Schema: 172.20.y.(10+x)
    parts = ip.split('.')
    y = int(parts[2])
    x = int(parts[3]) - 10
    return x, y

def coords_to_ip(x, y):
    return f"172.20.{y}.{10+x}"

    subprocess.run(cmd, shell=True, check=False) # check=False to avoid crashing if iptables fails (e.g. privs)

def apply_rate_limiting():
    """
    Limits SSH connections to prevent Brute-Force DoS.
    Limit: 10 connections per minute, burst of 5.
    """
    print("Applying Network Rate Limiting (SSH)...")
    try:
        # 1. Allow Established Connections (So we don't kill ourselves)
        run_cmd("iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT")
        
        # 2. Rate Limit NEW SSH Connections
        # Uses 'hashlimit' to track per-source-IP
        limit_cmd = "iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m hashlimit --hashlimit-name ssh_throttle --hashlimit-upto 10/min --hashlimit-burst 5 --hashlimit-mode srcip -j ACCEPT"
        run_cmd(limit_cmd)
        
        # 3. Drop Excessive NEW SSH Connections
        run_cmd("iptables -A INPUT -p tcp --dport 22 -m state --state NEW -j DROP")
        
    except Exception as e:
        print(f"Failed to apply rate limiting: {e}")

def setup_firewall(my_x, my_y):
    print(f"Setting up Directional Firewall for Node ({my_x}, {my_y})...")
    
    # 1. Flush existing rules for cleanliness (optional, potentially dangerous if other rules exist)
    # run_cmd("iptables -F") 
    
    for i, (dx, dy) in enumerate(DIRECTIONS):
        port = BASE_PORT + i
        nx, ny = my_x + dx, my_y + dy
        
        # Check if neighbor is valid
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbor_ip = coords_to_ip(nx, ny)
            print(f"  Port {port} (Dir {i}) <- Allow {neighbor_ip}")
            
            # Allow Neighbor
            run_cmd(f"iptables -A INPUT -p tcp --dport {port} -s {neighbor_ip} -j ACCEPT")
        
        # Drop everything else for this port
        run_cmd(f"iptables -A INPUT -p tcp --dport {port} -j DROP")

def start_listener(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', port))
    s.listen(1)
    # print(f"Listening on port {port}...")
    while True:
        conn, addr = s.accept()
        conn.close() # Dummy: "Ping! You hit the wall."

if __name__ == "__main__":
    try:
        my_ip = get_my_ip()
        my_x, my_y = ip_to_coords(my_ip)
        
        # Setup Rules
        apply_rate_limiting()
        setup_firewall(my_x, my_y)
        
        # NOTE: We do NOT start listeners here. 
        # The Gladiator (User AI) must bind to ports 9000-9007 to detect attacks.
        # If they don't, the port is closed (RST), but the firewall still protects from wrong directions.
            
        print("Directional Defense Rules Applied.")
        
    except Exception as e:
        print(f"Error establishing defense: {e}")
        # Identify logic for fallback? or just fail.
