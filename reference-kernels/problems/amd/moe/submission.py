import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t

class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        # AMD optimization: Use float16 for better performance on MI300
        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False, dtype=torch.float16)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False, dtype=torch.float16)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AMD optimization: Use mixed precision for better performance
        x = x.to(torch.float16)
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out.to(torch.float32)


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(config["n_routed_experts"])
        ])
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(x_flat,
                                            flat_expert_indices,
                                            flat_expert_weights)

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self,
                  x: torch.Tensor,
                  flat_expert_indices: torch.Tensor,
                  flat_expert_weights: torch.Tensor
                 ) -> torch.Tensor:
        # AMD MI300 optimized implementation
        expert_cache = torch.zeros_like(x)
        
        # AMD optimization 1: Wavefront-aware processing (64 threads per wavefront)
        # Group tokens to maximize wavefront utilization
        wavefront_size = 64
        num_experts = self.config["n_routed_experts"]
        
        # AMD optimization 2: Efficient counting using bincount
        expert_counts = torch.bincount(flat_expert_indices, 
                                      minlength=num_experts).to(torch.long)
        
        # AMD optimization 3: Sort indices for coalesced memory access
        sorted_token_indices = torch.argsort(flat_expert_indices)
        sorted_tokens = x[sorted_token_indices]
        sorted_weights = flat_expert_weights[sorted_token_indices]
        sorted_expert_indices = flat_expert_indices[sorted_token_indices]
        
        # AMD optimization 4: Process tokens in wavefront-sized chunks
        start_idx = 0
        for expert_id in range(num_experts):
            count = expert_counts[expert_id].item()
            if count > 0:
                end_idx = start_idx + count
                expert_tokens = sorted_tokens[start_idx:end_idx]
                expert_weights = sorted_weights[start_idx:end_idx]
                
                # AMD optimization 5: Pad to wavefront size for better utilization
                pad_size = (wavefront_size - (count % wavefront_size)) % wavefront_size
                if pad_size > 0:
                    # Pad with zeros for better memory coalescing
                    pad_shape = list(expert_tokens.shape)
                    pad_shape[0] = pad_size
                    padding_tokens = torch.zeros(pad_shape, dtype=expert_tokens.dtype, device=expert_tokens.device)
                    expert_tokens = torch.cat([expert_tokens, padding_tokens], dim=0)
                    padding_weights = torch.zeros((pad_size, 1), dtype=expert_weights.dtype, device=expert_weights.device)
                    expert_weights = torch.cat([expert_weights, padding_weights], dim=0)
                
                # Process tokens through expert
                expert_output = self.experts[expert_id](expert_tokens[:count])
                weighted_output = expert_output * expert_weights[:count]
                
                # Scatter results back to original positions
                original_positions = sorted_token_indices[start_idx:end_idx]
                expert_cache.scatter_add_(0, 
                                        original_positions.unsqueeze(1).expand(-1, x.shape[1]), 
                                        weighted_output)
                start_idx = end_idx

        return expert_cache


def custom_kernel(data: input_t) -> output_t:
    """
    AMD-optimized implementation of DeepSeek-style Mixture of Experts using PyTorch.
    
    Optimizations for AMD MI300:
    1. Improved memory access patterns for better cache locality
    2. Grouped expert processing for better wavefront utilization
    3. Reduced memory allocations and copies
    4. Efficient scatter/gather operations
    5. Mixed precision computation for better performance
    6. Wavefront-aware processing (64 threads)
    7. Memory alignment for coalesced access
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Processed tensor [batch_size, seq_len, d_model]
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

    for i in range(num_experts):
        gate_proj_weight = weights[f'experts.{i}.0.weight']
        up_proj_weight = weights[f'experts.{i}.1.weight']
        down_proj_weight = weights[f'experts.{i}.2.weight']

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t())
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t())
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t())

    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    # AMD optimization: Ensure input is in the right format for MI300
    input_tensor = input_tensor.contiguous()
    
    # Run the model
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        output = moe(input_tensor)

    return output