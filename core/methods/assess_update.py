import torch
import torch.nn.functional as F
from typing import Any, Dict, List

def assess_transient_branch(lora_wrapper: Any, transient_name: str, existing_branches: List[str], threshold: float = 0.3) -> Dict[str, Any]:
    """
    Assess the transient branch by decomposing its weights into shared and isolated subspaces.
    Returns a dict with decision to spawn or merge.
    """
    transient_vec = lora_wrapper.get_adapter_vector(transient_name, detach=True).float()
    if transient_vec.numel() == 0:
        return {"action": "merge", "target": existing_branches[0] if existing_branches else None, "isolated_energy_ratio": 0.0}
        
    total_energy = transient_vec.norm().item() ** 2
    if total_energy == 0:
        return {"action": "merge", "target": existing_branches[0] if existing_branches else None, "isolated_energy_ratio": 0.0}

    # Project transient_vec onto the subspace spanned by existing_branches
    # We use Gram-Schmidt or just simple projection if we assume branches are somewhat orthogonal
    # For simplicity and robustness, we project onto each branch and subtract
    residual = transient_vec.clone()
    
    best_sim = -1.0
    best_match = None
    
    for branch in existing_branches:
        branch_vec = lora_wrapper.get_adapter_vector(branch, detach=True).to(residual.device).float()
        if branch_vec.numel() == 0:
            continue
            
        branch_norm = branch_vec.norm().item()
        if branch_norm < 1e-6:
            # Found an empty branch, we can just use it!
            return {
                "action": "merge",
                "target": branch,
                "isolated_energy_ratio": 0.0,
                "best_sim": 1.0
            }
            
        # Calculate similarity for merging decision
        sim = F.cosine_similarity(transient_vec.unsqueeze(0), branch_vec.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_match = branch
            
        # Project out
        unit_branch = branch_vec / branch_norm
        proj = torch.dot(residual, unit_branch)
        residual = residual - proj * unit_branch
            
    isolated_energy = residual.norm().item() ** 2
    isolated_energy_ratio = isolated_energy / total_energy
    
    if isolated_energy_ratio > threshold:
        action = "spawn"
    else:
        action = "merge"
        
    return {
        "action": action,
        "target": best_match,
        "isolated_energy_ratio": isolated_energy_ratio,
        "best_sim": best_sim
    }
