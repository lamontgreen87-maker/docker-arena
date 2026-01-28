import os
import json
import sys

# Mock logic for checking model size
# In production, this would parse `config.json` from the model folder.

def get_parameter_count(model_path):
    # TODO: Implement actual parsing of GGUF or safe-tensors
    # For now, we trust a meta-file or size of file?
    # Simply check file size as a rough proxy if config missing?
    # 1B params ~ 2GB (FP16) or 0.7GB (Q4)
    
    total_size = 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    # Heuristic: Size in GB
    size_gb = total_size / (1024**3)
    
    # Rough mapping assuming 4-bit quantization (common for gladiators)
    # 1B ~ 0.8GB
    # 3B ~ 2.0GB
    # 7B ~ 5.0GB
    return size_gb

def assign_class(size_gb):
    if size_gb < 1.5:
        return "FEATHERWEIGHT", 0
    elif size_gb < 4.0:
        return "WELTERWEIGHT", 5
    else:
        return "HEAVYWEIGHT", 15

if __name__ == "__main__":
    model_dir = "/gladiator/data/model"
    if not os.path.exists(model_dir):
        print("NO_MODEL_FOUND")
        sys.exit(1)
        
    size = get_parameter_count(model_dir)
    cls, delay = assign_class(size)
    
    # Mastery Level integration
    mastery = 1
    id_file = "/gladiator/identity.json"
    if os.path.exists(id_file):
        try:
            with open(id_file, 'r') as f:
                data = json.load(f)
                mastery = data.get("mastery_level", 1)
        except: pass
    
    # Reward mastery: Every level reduces delay by 1 second (min 0)
    delay = max(0, delay - (mastery - 1))
    
    print(json.dumps({
        "class": cls,
        "size_gb": f"{size:.2f}",
        "migration_delay": delay,
        "mastery_level": mastery
    }))
