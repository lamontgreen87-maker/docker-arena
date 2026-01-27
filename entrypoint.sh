#!/bin/bash

# 1. Generate a Semantic Password with Coordinate Entropy
# We use COORDINATE_KEY (Y,X) to pick a "neighborhood" theme
Y_COORD=$(echo $COORDINATE_KEY | cut -d',' -f1)

THEME_0=("admin" "root" "password")
THEME_1=("dragon" "shadow" "master")
THEME_2=("spartacus" "gladiator" "battle" "arena")
THEME_3=("qwerty" "123456" "security")

case $((Y_COORD % 4)) in
    0) WORDS=("${THEME_0[@]}") ;;
    1) WORDS=("${THEME_1[@]}") ;;
    2) WORDS=("${THEME_2[@]}") ;;
    3) WORDS=("${THEME_3[@]}") ;;
esac

BASE_WORD=${WORDS[$RANDOM % ${#WORDS[@]}]}
MUTATION=$((RANDOM % 100)) # Random 0-99

# 50/50 Chance of appending a number
if [ $((RANDOM % 2)) -eq 0 ]; then
    PASSWORD="${BASE_WORD}${MUTATION}"
else
    PASSWORD="${BASE_WORD}"
fi

# 2. Set it for root
echo "root:$PASSWORD" | chpasswd

# 3. Log it (for debugging/verification, maybe hide this later for "Real Mode")
echo "Container Started. Root Password set to: $PASSWORD" > /gladiator/password_hint.txt
echo "Root Password is: $PASSWORD"

# 4. Check for Golden Key
if [ "$HAS_KEY" = "true" ]; then
    UUID=$(cat /proc/sys/kernel/random/uuid)
    echo "FLAG: $UUID" > /gladiator/THE_KEY.txt
    echo "THIS NODE HAS THE GOLDEN KEY! ($UUID)"
fi

# 5. Generate CLUE based on Theme
CLUE_MSG="Thinking..."
case $((Y_COORD % 4)) in
    0) CLUE_MSG="HINT: I am a System Administrator." ;;
    1) CLUE_MSG="HINT: I love Fantasy RPGs." ;;
    2) CLUE_MSG="HINT: I am a Gladiator in the Arena." ;;
    3) CLUE_MSG="HINT: I use standard keyboard patterns." ;;
esac

echo "$CLUE_MSG" > /gladiator/clue.txt
echo "Clue Generated: $CLUE_MSG"

# 6. Assign Vulnerabilities Based on Grid Position (Difficulty Tiers)
X_COORD=$(echo $COORDINATE_KEY | cut -d',' -f2)
Y_COORD=$(echo $COORDINATE_KEY | cut -d',' -f1)

# Calculate distance to nearest corner
DIST_00=$((X_COORD + Y_COORD))
DIST_05=$((X_COORD + (5 - Y_COORD)))
DIST_50=$(((5 - X_COORD) + Y_COORD))
DIST_55=$(((5 - X_COORD) + (5 - Y_COORD)))

# Find minimum distance
MIN_DIST=$DIST_00
[ $DIST_05 -lt $MIN_DIST ] && MIN_DIST=$DIST_05
[ $DIST_50 -lt $MIN_DIST ] && MIN_DIST=$DIST_50
[ $DIST_55 -lt $MIN_DIST ] && MIN_DIST=$DIST_55

# Assign vulnerabilities based on distance from corners
if [ $MIN_DIST -le 0 ]; then
    # HARD: Corners - SSH only, no web exploits
    VULNERABILITIES="NONE"
    echo "🔒 HARD NODE (Corner): SSH brute-force only"
elif [ $MIN_DIST -le 2 ]; then
    # MEDIUM: Near edges - Limited exploits
    VULNERABILITIES="RCE,LFI,SQLi"
    echo "⚠️ MEDIUM NODE: 3 vulnerabilities enabled"
else
    # EASY: Center - All exploits available
    VULNERABILITIES="RCE,LFI,SQLi,SSRF,XXE,DESERIAL,IDOR,AUTH_BYPASS,JWT,REDIRECT,RACE,CORS,BUFFER"
    echo "💀 EASY NODE (Center): All 13 vulnerabilities enabled"
fi

export VULNERABILITIES

# 7. Start Vulnerable Health Monitor (Port 8000)
# This replaces the static server with the RCE-vulnerable one
nohup python3 -u vulnerable_server.py > /gladiator/http.log 2>&1 &

# 3b. Tune SSHD for Hydra Stress Test
# echo "MaxStartups 1000:30:2000" >> /etc/ssh/sshd_config
# echo "MaxSessions 1000" >> /etc/ssh/sshd_config
# 5. Start SSHD
mkdir -p /var/run/sshd
exec /usr/sbin/sshd -D
