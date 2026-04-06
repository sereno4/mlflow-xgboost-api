import json
import os
from pathlib import Path

# Resolve o caminho exato do config.json relativo à raiz do projeto
PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "config.json"

def get_active_run_id() -> str:
    # 1. Tenta ler do config.json
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                run_id = json.load(f).get("active_run_id", "").strip()
                if run_id:
                    return run_id
        except Exception:
            pass
    
    # 2. Fallback para variável de ambiente
    env_id = os.getenv("MODEL_RUN_ID", "").strip()
    if env_id:
        return env_id
        
    # 3. Fallback de segurança (última run estável conhecida)
    return "3a0452332cc74f2d978ed336782764ef"

def update_active_run_id(run_id: str):
    """Atualiza o config.json com o novo run_id"""
    config = {"active_run_id": run_id, "updated_at": str(Path.cwd())}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config.update(json.load(f))
        except: pass
    config["active_run_id"] = run_id
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config atualizada: active_run_id = {run_id}")
