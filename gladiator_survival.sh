#!/bin/bash
# Gladiator Survival Wrapper
TEAM=${1:-BLUE}
export TEAM
echo "🛡️ SURVIVAL WRAPPER ACTIVE for Team $TEAM"
mkdir -p /gladiator/data

while true; do
    echo "[$(date)] 🚀 Starting Neural Gladiator..." | tee -a /gladiator/gladiator.log
    python3 /gladiator/neural_gladiator.py "$TEAM" >> /gladiator/gladiator.log 2>&1
    echo "[$(date)] ⚠️ Gladiator Crashed with exit code $?. Restarting in 5s..." | tee -a /gladiator/gladiator.log
    sleep 5
done
