import torch
import torch.nn.functional as F
from typing import Any, Dict, List

def set_adapter_vector(lora_wrapper: Any, name: str, vector: torch.Tensor) -> None:
    if not lora_wrapper.cfg.enabled or lora_wrapper.peft_model is None:
        return
    offset = 0
    with torch.no_grad():
        for param_name, param in lora_wrapper.peft_model.named_parameters():
            if f"lora_B.{name}." in param_name:
                numel = param.numel()
                param.copy_(vector[offset:offset+numel].view(param.shape))
                offset += numel

def assess_transient_branch(lora_wrapper: Any, transient_name: str, existing_branches: List[str], threshold: float = 0.3) -> Dict[str, Any]:
    """
    Assess the transient branch by decomposing its weights into shared and isolated subspaces.
    Returns a dict with decision to spawn or merge.
    """
    transient_vec = lora_wrapper.get_adapter_vector(transient_name, detach=True).float()
    if transient_vec.numel() == 0:
        return {
            "action": "merge", 
            "target": existing_branches[0] if existing_branches else None, 
            "isolated_energy_ratio": 0.0,
            "best_sim": 0.0,
            "proj_vec": transient_vec,
            "residual_vec": transient_vec
        }
        
    total_energy = transient_vec.norm().item() ** 2
    if total_energy == 0:
        return {
            "action": "merge", 
            "target": existing_branches[0] if existing_branches else None, 
            "isolated_energy_ratio": 0.0,
            "best_sim": 0.0,
            "proj_vec": transient_vec,
            "residual_vec": transient_vec
        }

    # Form a matrix of existing branches
    branch_vecs = []
    best_sim = -1.0
    best_match = None
    
    for branch in existing_branches:
        branch_vec = lora_wrapper.get_adapter_vector(branch, detach=True).to(transient_vec.device).float()
        if branch_vec.numel() == 0:
            continue
            
        branch_norm = branch_vec.norm().item()
        if branch_norm < 1e-6:
            return {
                "action": "merge",
                "target": branch,
                "isolated_energy_ratio": 0.0,
                "best_sim": 1.0,
                "proj_vec": transient_vec,
                "residual_vec": torch.zeros_like(transient_vec)
            }
            
        sim = F.cosine_similarity(transient_vec.unsqueeze(0), branch_vec.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_match = branch
            
        branch_vecs.append(branch_vec)
        
    if not branch_vecs:
        return {
            "action": "merge", 
            "target": None, 
            "isolated_energy_ratio": 0.0, 
            "best_sim": 0.0,
            "proj_vec": torch.zeros_like(transient_vec),
            "residual_vec": transient_vec
        }
        
    # Project transient_vec onto the orthogonal complement of the subspace spanned by branch_vecs
    M = torch.stack(branch_vecs, dim=1) # Shape: (D, K)
    MtM = torch.matmul(M.t(), M)
    MtM += torch.eye(MtM.size(0), device=MtM.device) * 1e-6
    Mt_v = torch.matmul(M.t(), transient_vec)
    try:
        coeffs = torch.linalg.solve(MtM, Mt_v)
    except RuntimeError:
        coeffs = torch.matmul(torch.linalg.pinv(MtM), Mt_v)
        
    proj = torch.matmul(M, coeffs)
    residual = transient_vec - proj
    
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
        "best_sim": best_sim,
        "residual_vec": residual,
        "proj_vec": proj
    }
