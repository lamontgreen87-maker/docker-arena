
import json

def test_mapping(ip, expected_x, expected_y):
    parts = ip.split('.')
    # System Convention: Octet 2 is Row (Y), Octet 3 - 10 is Col (X)
    # Orchestrator/Smart Gladiator: y = octet[2], x = octet[3] - 10
    y = int(parts[2])
    x = int(parts[3]) - 10
    
    print(f"Testing {ip}:")
    print(f"  Expected: ({expected_x}, {expected_y})")
    print(f"  Parsed:   ({x}, {y})")
    
    return x == expected_x and y == expected_y

# Test cases based on docker-compose.yml
test_cases = [
    ("172.20.0.10", 0, 0),
    ("172.20.0.11", 1, 0),
    ("172.20.1.10", 0, 1),
    ("172.20.3.12", 2, 3), # The "KEY" node
]

all_passed = True
for ip, ex, ey in test_cases:
    if not test_mapping(ip, ex, ey):
        print("  FAILED!")
        all_passed = False
    else:
        print("  PASSED")

if all_passed:
    print("\nSUCCESS: IP to Coordinate mapping is consistent with docker-compose.yml")
else:
    print("\nFAILURE: Mapping inconsistency detected!")
