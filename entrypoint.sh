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

# 3. Start Directional Defense (Background)
# if [ -f "/gladiator/defense.sh" ]; then
#     /gladiator/defense.sh &
# fi

# 3b. Tune SSHD for Hydra Stress Test
# echo "MaxStartups 1000:30:2000" >> /etc/ssh/sshd_config
# echo "MaxSessions 1000" >> /etc/ssh/sshd_config
# 5. Start SSHD
mkdir -p /var/run/sshd
exec /usr/sbin/sshd -D
