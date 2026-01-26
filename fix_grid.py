import docker
import subprocess

client = docker.from_env()
containers = client.containers.list()

for c in containers:
    if "arena_" in c.name and "orchestrator" not in c.name:
        print(f"Patching {c.name}...")
        try:
            # 1. Force PermitRootLogin via dedicated config file (overrides main config)
            c.exec_run("bash -c \"echo 'PermitRootLogin yes' > /etc/ssh/sshd_config.d/hack.conf\"")
            c.exec_run("bash -c \"echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config.d/hack.conf\"")
            
            # 2. Reset Password to '123456' (Top of list)
            c.exec_run("bash -c \"echo 'root:123456' | chpasswd\"")
            
            # 3. Restart SSHD
            c.exec_run("service ssh restart")
            
            print(f"  > Patched & Restarted SSHD")
        except Exception as e:
            print(f"  > Failed: {e}")
