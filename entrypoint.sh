#!/bin/bash

# 1. Pick a random password
PASSWORD=$(shuf -n 1 /gladiator/passwords.txt)

# 2. Set it for root
echo "root:$PASSWORD" | chpasswd

# 3. Log it (for debugging/verification, maybe hide this later for "Real Mode")
echo "Container Started. Root Password set to: $PASSWORD" > /gladiator/password_hint.txt
echo "Root Password is: $PASSWORD"

# 4. Start Directional Defense (Background)
python3 /gladiator/directional_defense.py &

# 5. Start SSHD
exec /usr/sbin/sshd -D
