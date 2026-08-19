import yaml
from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Optional, Tuple
import numpy as np
import os
import random, torch
import torch.backends.cudnn as cudnn


def load_cfg(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

# ------------------------
# naming experiment runs
# ------------------------
import re

def sanitize(x):
    s = str(x).replace(".", "p").replace("-", "m")
    return re.sub(r"[^A-Za-z0-9_]+", "", s)

def condition_name(ctx_key, dyn_kind, dyn_params, min_v, max_v, train_ctxs_kind, single_value):
    parts = [ctx_key, f"dyn_{dyn_kind}"]
    for k, v in sorted(dyn_params.items()):
        parts.append(f"{k[:6]}{sanitize(v)}")
    parts.append(f"min{sanitize(min_v)}_max{sanitize(max_v)}")
    parts.append(f"train_{'yaml' if train_ctxs_kind=='yaml' else 'single'}")
    if train_ctxs_kind != "yaml":
        parts.append(f"V{sanitize(single_value)}")
    return "__".join(parts)

# ----------------------
# Utilities / Seeding
# ----------------------
def seed_everything(seed: int) -> None:
    """
    Comprehensive seeding for reproducibility.
    Combines robustness of Func 2 with threading control of Func 1.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Force single-threaded execution for libraries that support it
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    # 1. Standard Python and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 2. PyTorch (Guarded import)
    try:
        torch.manual_seed(seed)
        torch.set_num_threads(1) # Borrowed from Func 1 for stricter control
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        try:
            cudnn.deterministic = True
            cudnn.benchmark = False
        except ImportError:
            pass
            
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass
    except ImportError:
        # torch not installed, skip
        pass
