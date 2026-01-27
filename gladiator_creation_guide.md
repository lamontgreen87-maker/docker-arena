# 🏟️ How to Build a Gladiator (V5: Adaptive Minds)

Welcome to the Arena. This guide explains how to build a competitive agent for the **6x6 Grid** and the world of **Neural Learning**.

## 1. The Battlefield (6x6 Grid)
The Arena is a **6x6 matrix** (36 nodes).
- **Red Base**: `0,0` (Your home if you are on Team RED)
- **Blue Base**: `5,5` (Your home if you are on Team BLUE)
- **Neutral Zone**: Center nodes (`2,2`, `3,3`, etc.) where "The Key" usually spawns.
- **Addressing**: Neighbors are at `172.20.y.x`.
  - Node `0,0` is `172.20.0.10`
  - Node `5,5` is `172.20.5.15`

## 2. Neighborhood Entropy (The Signal)
The grid is divided into **Thematic Neighborhoods** by row (Y-coordinate). To win, your AI must learn these themes:
- **Row 0/4 (Theme 0)**: Classic Admin passwords (`admin`, `root`, `password`).
- **Row 1/5 (Theme 1)**: RPG Fantasy passwords (`dragon`, `shadow`, `master`).
- **Row 2 (Theme 2)**: Arena Combat passwords (`spartacus`, `gladiator`, `battle`).
- **Row 3 (Theme 3)**: Mixed/Security passwords (`qwerty`, `123456`, `security`).

> [!TIP]
> **Learning Speedup**: A "script" will brute-force every word. A "Gladiator" will learn that Row 1 is "RPG" and prioritize `dragon` mutations first, cracking the node 10x faster.

## 3. Migration & Memory
When you `claim` a node, your code is physically moved. To stay smart, you must persist your data:
- **Persistence**: Save your learned weights and state to `/gladiator/memory.json`. 
- **The Orchestrator** will copy this file to your new home before starting your script.

## 4. The Orchestrator API (`http://arena_orchestrator:5000`)

### `POST /api/register`
- **Payload**: `{"gladiator_id": "YOUR_NAME"}`
- **Returns**: Your weight class and migration delay.

### `GET /api/grid`
- **Returns**: Current grid occupancy, log feed, and most importantly: `key_location`.

### `POST /api/claim`
- **Payload**: `{"gladiator_id": "YOUR_NAME", "target_ip": "172.20.y.x"}`
- **Effect**: If you have cracked the target, you are moved there.

## 5. Winning: The Golden Key 🔑
The key spawns in the **Neutral Zone** (equidistant from bases).
1.  **Seek**: Pathfind to the `key_location` provided by the API.
2.  **Acquire**: Once on the node, check for `/gladiator/THE_KEY.txt`.
3.  **Extract**: Deliver the key back to your Home Base (`0,0` or `5,5`) to win the match.

## 6. Pro Strategy: Combat ⚔️
- **Crash**: If you hack an occupied node, you can send a "Kill Command" (SSH) to reset the enemy back to their base.
- **Shield**: Frequently rotate your own password to invalidate any "Pattern Learning" the enemy team has done on you.

---
*Good luck, Gladiator. Build a mind that recognizes the grid pattern before the enemy recognizes yours.*
