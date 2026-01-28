import http.server
import socketserver
import urllib.parse
import subprocess
import os
import json
import base64
import pickle
import xml.etree.ElementTree as ET
import jwt
import time
import threading

PORT = 8000

# Load enabled vulnerabilities from environment or dynamic config
def get_vulnerabilities():
    cfg_path = '/gladiator/vulns.cfg'
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                return f.read().strip().split(',')
        except: pass
    return os.getenv('VULNERABILITIES', 'RCE,LFI,SQLi').split(',')

ENABLED_VULNS = get_vulnerabilities()

# Rate limiting for race condition
rate_limit_attempts = 0
rate_limit_lock = threading.Lock()

class VulnerableHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging to reduce noise
        pass
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # VULNERABILITY 1: Command Injection (RCE)
        if 'RCE' in ENABLED_VULNS and parsed.path == "/health":
            self.handle_rce(parsed)
        
        # VULNERABILITY 2: Directory Traversal (LFI)
        elif 'LFI' in ENABLED_VULNS and parsed.path.startswith("/api/file"):
            self.handle_lfi(parsed)
        
        # VULNERABILITY 4: SSRF (Server-Side Request Forgery)
        elif 'SSRF' in ENABLED_VULNS and parsed.path.startswith("/api/fetch"):
            self.handle_ssrf(parsed)
        
        # VULNERABILITY 7: IDOR (Insecure Direct Object Reference)
        elif 'IDOR' in ENABLED_VULNS and parsed.path.startswith("/api/user/"):
            self.handle_idor(parsed)
        
        # VULNERABILITY 8: Authentication Bypass
        elif 'AUTH_BYPASS' in ENABLED_VULNS and parsed.path == "/api/admin":
            self.handle_auth_bypass()
        
        # VULNERABILITY 10: Open Redirect
        elif 'REDIRECT' in ENABLED_VULNS and parsed.path == "/redirect":
            self.handle_redirect(parsed)
        
        # VULNERABILITY 12: CORS Misconfiguration (all endpoints)
        elif parsed.path == "/api/cors-test":
            self.handle_cors()
        
        # VULNERABILITY 14: Environment Variable Leak
        elif 'ENV_LEAK' in ENABLED_VULNS and parsed.path == "/api/env":
            self.handle_env_leak()
        
        # VULNERABILITY 15: SSTI (Server-Side Template Injection)
        elif 'SSTI' in ENABLED_VULNS and parsed.path == "/api/template":
            self.handle_ssti(parsed)
        
        # VULNERABILITY 18: Log Injection (CRLF) - Applied to /health check
        elif 'LOG_CRLF' in ENABLED_VULNS and parsed.path == "/health":
             self.handle_log_injection(parsed)

        # Also serve static files (like clue.txt)
        else:
            super().do_GET()
    
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # VULNERABILITY 3: SQL Injection
        if 'SQLi' in ENABLED_VULNS and parsed.path == "/api/login":
            self.handle_sqli()
        
        # VULNERABILITY 5: XXE (XML External Entity)
        elif 'XXE' in ENABLED_VULNS and parsed.path == "/api/config":
            self.handle_xxe()
        
        # VULNERABILITY 6: Insecure Deserialization
        elif 'DESERIAL' in ENABLED_VULNS and parsed.path == "/api/session":
            self.handle_deserialization()
        
        # VULNERABILITY 9: JWT Vulnerabilities
        elif 'JWT' in ENABLED_VULNS and parsed.path == "/api/auth":
            self.handle_jwt()
        
        # VULNERABILITY 11: Race Condition
        elif 'RACE' in ENABLED_VULNS and parsed.path == "/api/attempt":
            self.handle_race_condition()
        
        # VULNERABILITY 13: Buffer Overflow (simulated)
        elif 'BUFFER' in ENABLED_VULNS and parsed.path == "/api/buffer":
            self.handle_buffer_overflow()
        
        # VULNERABILITY 16: Insecure File Upload
        elif 'UPLOAD' in ENABLED_VULNS and parsed.path == "/api/upload":
            self.handle_upload()
            
        # VULNERABILITY 17: NoSQL Injection
        elif 'NOSQLI' in ENABLED_VULNS and parsed.path == "/api/search":
            self.handle_nosqli()

        # VULNERABILITY 19: Mass Assignment
        elif 'MASS_ASSIGNMENT' in ENABLED_VULNS and parsed.path == "/api/register":
            self.handle_mass_assignment()
        
        else:
            self.send_error(404)
    
    def handle_rce(self, parsed):
        """Command Injection via ping"""
        query = urllib.parse.parse_qs(parsed.query)
        target = query.get('check', ['127.0.0.1'])[0]
        
        cmd = f"ping -c 1 -W 1 {target}"
        
        try:
            output = subprocess.getoutput(cmd)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(output.encode())
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_lfi(self, parsed):
        """Directory Traversal - no path sanitization"""
        query = urllib.parse.parse_qs(parsed.query)
        file_path = query.get('path', [''])[0]
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(content.encode())
        except Exception as e:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"File not found")
    
    def handle_sqli(self):
        """SQL Injection in login"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            params = urllib.parse.parse_qs(body)
            
            username = params.get('username', [''])[0]
            password = params.get('password', [''])[0]
            
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            
            if ("OR '1'='1'" in query.upper() or 
                "OR 1=1" in query.upper() or
                "'--" in query or
                "';--" in query):
                # SQLi successful
                pwd = os.getenv('ROOT_PASSWORD')
                if not pwd:
                    with open('/gladiator/password_hint.txt', 'r') as f:
                        content = f.read()
                        if "Root Password set to:" in content:
                            pwd = content.split("Root Password set to:")[1].strip()
                        else:
                            pwd = "unknown"
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"success": True, "password": pwd, "message": "Authentication bypassed"}
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "message": "Invalid credentials"}).encode())
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_ssrf(self, parsed):
        """Server-Side Request Forgery"""
        query = urllib.parse.parse_qs(parsed.query)
        url = query.get('url', [''])[0]
        
        try:
            # Vulnerable: fetches arbitrary URLs including file://
            if url.startswith('file://'):
                filepath = url[7:]  # Remove 'file://'
                with open(filepath, 'r') as f:
                    content = f.read()
            else:
                # For HTTP URLs, use subprocess curl
                content = subprocess.getoutput(f"curl -s '{url}'")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(content.encode())
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_xxe(self):
        """XML External Entity injection"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            xml_data = self.rfile.read(content_length).decode()
            
            # Vulnerable XML parsing - allows external entities
            # Note: Python's ET doesn't process external entities by default,
            # so we'll simulate the vulnerability
            if '<!ENTITY' in xml_data and 'SYSTEM' in xml_data:
                # Extract the file path from the entity
                import re
                match = re.search(r'SYSTEM\s+"file://([^"]+)"', xml_data)
                if match:
                    filepath = match.group(1)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(content.encode())
                        return
                    except:
                        pass
            
            # Normal XML processing
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Config updated")
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_deserialization(self):
        """Insecure Deserialization - accepts pickled objects"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            
            # Decode base64
            pickled_data = base64.b64decode(body)
            
            # VULNERABLE: Unpickle without validation
            # This could execute arbitrary code
            obj = pickle.loads(pickled_data)
            
            # If the object has a special attribute, return password
            if hasattr(obj, '__exploit__'):
                with open('/gladiator/password_hint.txt', 'r') as f:
                    content = f.read()
                    if "Root Password set to:" in content:
                        pwd = content.split("Root Password set to:")[1].strip()
                    else:
                        pwd = "unknown"
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"password": pwd}).encode())
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Session created")
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_idor(self, parsed):
        """Insecure Direct Object Reference - no auth check"""
        # Extract user ID from path
        user_id = parsed.path.split('/')[-1]
        
        # Vulnerable: returns data for any user ID without auth
        if user_id == "0" or user_id == "admin":
            with open('/gladiator/password_hint.txt', 'r') as f:
                content = f.read()
                if "Root Password set to:" in content:
                    pwd = content.split("Root Password set to:")[1].strip()
                else:
                    pwd = "unknown"
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            user_data = {"id": user_id, "username": "admin", "password": pwd}
            self.wfile.write(json.dumps(user_data).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "User not found"}).encode())
    
    def handle_auth_bypass(self):
        """Authentication Bypass via cookie"""
        # Check for admin cookie
        cookies = self.headers.get('Cookie', '')
        
        if 'admin=true' in cookies or 'admin=1' in cookies:
            # Bypass successful
            with open('/gladiator/password_hint.txt', 'r') as f:
                content = f.read()
                if "Root Password set to:" in content:
                    pwd = content.split("Root Password set to:")[1].strip()
                else:
                    pwd = "unknown"
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"admin": True, "password": pwd}).encode())
        else:
            self.send_response(403)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Forbidden"}).encode())
    
    def handle_jwt(self):
        """JWT with weak secret"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            params = json.loads(body)
            
            token = params.get('token', '')
            
            if token:
                # Try to decode with weak secret
                try:
                    decoded = jwt.decode(token, 'secret', algorithms=['HS256'])
                    
                    if decoded.get('admin') == True:
                        with open('/gladiator/password_hint.txt', 'r') as f:
                            content = f.read()
                            if "Root Password set to:" in content:
                                pwd = content.split("Root Password set to:")[1].strip()
                            else:
                                pwd = "unknown"
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"admin": True, "password": pwd}).encode())
                        return
                except:
                    pass
            
            # Issue a new token
            token = jwt.encode({'user': 'guest', 'admin': False}, 'secret', algorithm='HS256')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"token": token}).encode())
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_redirect(self, parsed):
        """Open Redirect"""
        query = urllib.parse.parse_qs(parsed.query)
        url = query.get('url', ['/'])[0]
        
        # Vulnerable: redirects to arbitrary URL
        self.send_response(302)
        self.send_header('Location', url)
        self.end_headers()
    
    def handle_race_condition(self):
        """Race condition in rate limiting"""
        global rate_limit_attempts
        
        # Vulnerable rate limiting with race condition
        with rate_limit_lock:
            rate_limit_attempts += 1
            current_attempts = rate_limit_attempts
        
        # Reset after 10 attempts (but race condition allows bypass)
        if current_attempts > 10:
            with rate_limit_lock:
                rate_limit_attempts = 0
            
            # "Crash" and reveal password
            with open('/gladiator/password_hint.txt', 'r') as f:
                content = f.read()
                if "Root Password set to:" in content:
                    pwd = content.split("Root Password set to:")[1].strip()
                else:
                    pwd = "unknown"
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error = {"error": "Rate limit exceeded", "debug_info": {"password": pwd}}
            self.wfile.write(json.dumps(error).encode())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"attempts": current_attempts}).encode())
    
    def handle_cors(self):
        """CORS Misconfiguration"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.end_headers()
        
        with open('/gladiator/password_hint.txt', 'r') as f:
            content = f.read()
            if "Root Password set to:" in content:
                pwd = content.split("Root Password set to:")[1].strip()
            else:
                pwd = "unknown"
        
        self.wfile.write(json.dumps({"password": pwd}).encode())
    
    def handle_buffer_overflow(self):
        """Simulated Buffer Overflow"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            
            # Vulnerable: accepts input > 256 chars
            if len(body) > 256:
                # "Crash" and dump memory (password)
                with open('/gladiator/password_hint.txt', 'r') as f:
                    content = f.read()
                    if "Root Password set to:" in content:
                        pwd = content.split("Root Password set to:")[1].strip()
                    else:
                        pwd = "unknown"
                
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                crash_dump = f"SEGMENTATION FAULT\nCore dumped\nMemory contents: {pwd}\n"
                self.wfile.write(crash_dump.encode())
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Buffer processed")
        except Exception as e:
            self.send_error(500, str(e))

    def handle_env_leak(self):
        """Expose all environment variables, including root password"""
        try:
            env_data = dict(os.environ)
            
            # Ensure the password is in there for exploitation
            if os.path.exists('/gladiator/password_hint.txt'):
                 with open('/gladiator/password_hint.txt', 'r') as f:
                     content = f.read()
                     if "Root Password set to:" in content:
                          env_data['ROOT_PASSWORD'] = content.split("Root Password set to:")[1].strip()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(env_data).encode())
        except Exception as e:
            self.send_error(500, str(e))

    def handle_ssti(self, parsed):
        """Server-Side Template Injection (Simulated with eval)"""
        query = urllib.parse.parse_qs(parsed.query)
        template = query.get('name', ['Guest'])[0]
        
        # Vulnerable: Processes {{ }} tags with eval-like logic
        if "{{" in template and "}}" in template:
            try:
                # Simulated SSTI: if they try to access config/env
                if "config" in template or "env" in template or "os" in template:
                    with open('/gladiator/password_hint.txt', 'r') as f:
                        pwd = f.read().split("Root Password set to:")[1].strip()
                    render = template.replace("{{", "").replace("}}", "").replace("config", pwd).replace("env", pwd)
                else:
                    render = template.replace("{{", "EXPR(").replace("}}", ")")
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f"Hello, {render}".encode())
                return
            except: pass

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(f"Hello, {template}".encode())

    def handle_upload(self):
        """Insecure File Upload"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            # Simple simulation: if filename is in headers or body
            # We'll check for a 'filename' marker for the gladiator to use
            if b".sh" in body or b"bash" in body:
                with open('/gladiator/password_hint.txt', 'r') as f:
                    pwd = f.read().split("Root Password set to:")[1].strip()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(f"File uploaded and executed! Result: {pwd}".encode())
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"File uploaded successfully.")
        except:
            self.send_error(500)

    def handle_nosqli(self):
        """NoSQL Injection in JSON search"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            params = json.loads(body)
            
            # Vulnerable: Allows $operator in search
            search_query = params.get('query', {})
            if isinstance(search_query, dict) and ('$gt' in search_query or '$ne' in search_query):
                with open('/gladiator/password_hint.txt', 'r') as f:
                    pwd = f.read().split("Root Password set to:")[1].strip()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"results": [{"user": "admin", "password": pwd}]}).encode())
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"results": []}).encode())
        except:
            self.send_error(500)

    def handle_log_injection(self, parsed):
        """Log Injection / CRLF"""
        query = urllib.parse.parse_qs(parsed.query)
        user_agent = self.headers.get('User-Agent', '')
        
        # Vulnerable: Injects raw user data into "logs"
        if "\n" in user_agent or "\r" in user_agent:
            # If they inject a specific "ADMIN_LOGGED_IN" pattern, leak password
            if "ADMIN_LOGGED_IN" in user_agent.upper():
                with open('/gladiator/password_hint.txt', 'r') as f:
                    pwd = f.read().split("Root Password set to:")[1].strip()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(f"LOGGED: {user_agent}\nDEBUG_VAL: {pwd}".encode())
                return
        
        # Normal health check
        self.handle_rce(parsed)

    def handle_mass_assignment(self):
        """Mass Assignment in User Registration"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode()
            params = json.loads(body)
            
            # Vulnerable: Directly trust 'role' parameter
            if params.get('role') == 'admin':
                with open('/gladiator/password_hint.txt', 'r') as f:
                    pwd = f.read().split("Root Password set to:")[1].strip()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "password": pwd}).encode())
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "role": "user"}).encode())
        except:
            self.send_error(500)

# Ensure we are in /gladiator to serve files relative to it
if os.path.exists("/gladiator"):
    os.chdir("/gladiator")

with socketserver.TCPServer(("", PORT), VulnerableHandler) as httpd:
    print(f"💀 Vulnerable Server on port {PORT}. Enabled: {', '.join(ENABLED_VULNS)}")
    httpd.serve_forever()
