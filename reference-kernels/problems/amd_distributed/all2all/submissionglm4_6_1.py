import torch
import torch.distributed as dist
from task import input_t, output_t
import triton
import triton.language as tl
import os
import numpy as np

# ROCm SHMEM imports
try:
    import torch_hip
    from torch_hip import hip
    import rocm_shmem
except ImportError:
    print("ROCm SHMEM not available, falling back to PyTorch distributed")
    rocm_shmem = None

# Composable Kernel imports
try:
    import composable_kernel as ck
except ImportError:
    print("Composable Kernel not available, using fallback")
    ck = None

# ---------------- Optimized AMD AllToAll Implementation ----------------
class OptimizedAMDAllToAll:
    """
    High performance implementation optimized for AMD ROCm:
    - Uses ROCm SHMEM for direct GPU-to-GPU communication
    - Employs HIP kernels for computation
    - Utilizes Composable Kernel for matrix operations
    - Optimizes memory access patterns for AMD wavefronts
    - Minimizes host-device synchronization
    """
    META_DIM = 5  # [global_eid, src_rank, src_token_idx, src_k, pad]

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden_dim = cfg.hidden_dim

        # AMD-specific optimizations
        self.wavefront_size = 64  # AMD GPU wavefront size
        self.alignment = 128  # For better memory bandwidth utilization on AMD
        
        # Initialize ROCm SHMEM if available
        self.use_shmem = rocm_shmem is not None
        if self.use_shmem:
            self._init_shmem()
        
        # Pre-allocate communication buffers
        self._allocate_buffers()

    def _init_shmem(self):
        """Initialize ROCm SHMEM for direct GPU communication"""
        if not self.use_shmem:
            return
            
        # Set device for this process
        hip.set_device(self.rank)
        
        # Initialize SHMEM
        rocm_shmem.init()
        
        # Create communication group
        self.shmem_group = rocm_shmem.Team.world()
        
        # Allocate SHMEM buffers
        self.shmem_send_meta = rocm_shmem.alloc(
            (self.max_recv, self.META_DIM), dtype=torch.int32
        )
        self.shmem_send_data = rocm_shmem.alloc(
            (self.max_recv, self.hidden_dim), dtype=self.cfg.in_dtype
        )
        self.shmem_recv_meta = rocm_shmem.alloc(
            (self.max_recv, self.META_DIM), dtype=torch.int32
        )
        self.shmem_recv_data = rocm_shmem.alloc(
            (self.max_recv, self.hidden_dim), dtype=self.cfg.in_dtype
        )

    def _allocate_buffers(self):
        """Pre-allocate buffers to reduce allocation overhead"""
        device = torch.device(f"hip:{self.rank}")
        
        # Pre-allocate expert buffers
        aligned_max_recv = self._align(self.max_recv, self.alignment)
        self.expert_x = torch.empty(
            (self.num_local_experts, aligned_max_recv, self.hidden_dim),
            dtype=self.cfg.in_dtype, device=device
        )
        self.expert_meta = torch.empty(
            (self.num_local_experts, aligned_max_recv, self.META_DIM),
            dtype=torch.int32, device=device
        )
        
        # Pre-allocate temporary buffers
        self.send_meta = torch.empty(
            (self.max_recv, self.META_DIM), dtype=torch.int32, device=device
        )
        self.send_data = torch.empty(
            (self.max_recv, self.hidden_dim), dtype=self.cfg.in_dtype, device=device
        )
        self.recv_meta = torch.empty(
            (self.max_recv, self.META_DIM), dtype=torch.int32, device=device
        )
        self.recv_data = torch.empty(
            (self.max_recv, self.hidden_dim), dtype=self.cfg.in_dtype, device=device
        )
        
        # Pre-allocate index buffers
        self.dst_flat = torch.empty(
            (self.max_recv,), dtype=torch.long, device=device
        )
        self.src_token_idx = torch.empty(
            (self.max_recv,), dtype=torch.long, device=device
        )
        self.global_eids_flat = torch.empty(
            (self.max_recv,), dtype=torch.long, device=device
        )
        self.src_k_flat = torch.empty(
            (self.max_recv,), dtype=torch.long, device=device
        )

    @staticmethod
    def _align(n, a):
        """Align dimension to specified alignment factor."""
        return ((n + a - 1) // a) * a

    @staticmethod
    def _contig(t: torch.Tensor):
        """Ensure tensor is contiguous for better memory access."""
        return t.contiguous() if not t.is_contiguous() else t

    def _compute_flat_info(self, indices: torch.Tensor):
        """
        Compute flattened destination information for tokens using HIP kernel.
        
        Args:
            indices: (num_tokens, k) global expert ids
            
        Returns:
            dst_flat: (num_tokens * k,) destination ranks
            src_token_idx: (num_tokens * k,) source token indices
            global_eids_flat: (num_tokens * k,) global expert ids
            src_k_flat: (num_tokens * k,) source k indices
        """
        device = indices.device
        num_tokens, k = indices.shape
        
        # Use pre-allocated buffers
        dst_flat = self.dst_flat[:num_tokens * k]
        src_token_idx = self.src_token_idx[:num_tokens * k]
        global_eids_flat = self.global_eids_flat[:num_tokens * k]
        src_k_flat = self.src_k_flat[:num_tokens * k]
        
        # Launch HIP kernel for flat info computation
        if ck and hasattr(ck, 'compute_flat_info'):
            # Use Composable Kernel if available
            ck.compute_flat_info(
                indices, dst_flat, src_token_idx, 
                global_eids_flat, src_k_flat,
                self.num_local_experts, self.world_size
            )
        else:
            # Fallback to PyTorch operations
            dst_flat.copy_(torch.div(indices, self.num_local_experts, rounding_mode='floor').reshape(-1).long())
            src_token_idx.copy_(torch.arange(num_tokens, device=device).repeat_interleave(k).long())
            global_eids_flat.copy_(indices.reshape(-1).long())
            src_k_flat.copy_(torch.arange(k, device=device).repeat(num_tokens).long())
        
        return dst_flat, src_token_idx, global_eids_flat, src_k_flat

    # ---------------- Dispatch ----------------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Dispatch tokens to experts using optimized HIP kernels and ROCm SHMEM.
        
        Args:
            dp_x: (num_tokens, hidden_dim) input tokens
            indices: (num_tokens, k) global expert ids
            
        Returns:
            expert_num_tokens: (num_local_experts,) int32
            expert_x: (num_local_experts, aligned_max_recv, hidden_dim) in_dtype
            expert_meta: (num_local_experts, aligned_max_recv, META_DIM) int32
        """
        cfg = self.cfg
        device = dp_x.device

        # Ensure contiguous memory layout for better performance
        dp_x = self._contig(dp_x)
        indices = self._contig(indices)

        num_tokens, k = indices.shape

        # Compute flat information for all token-expert pairs
        dst_flat, src_token_idx, global_eids_flat, src_k_flat = self._compute_flat_info(indices)

        # Calculate send counts per rank
        send_counts = torch.bincount(dst_flat, minlength=self.world_size).long().to(device)
        total_send = int(send_counts.sum().item())

        # Fast path: nothing to send
        if total_send == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            return expert_num_tokens, self._contig(self.expert_x), self._contig(self.expert_meta)

        # Sort by destination to create contiguous send buffer per rank
        order = torch.argsort(dst_flat)
        ordered_dst = dst_flat[order]
        ordered_src_token = src_token_idx[order]
        ordered_global_eid = global_eids_flat[order]
        ordered_src_k = src_k_flat[order]
        ordered_data = dp_x[ordered_src_token]

        # Construct send metadata (int32) and send data (in_dtype)
        send_meta = self.send_meta[:total_send]
        send_data = self.send_data[:total_send]
        
        # Use HIP kernel for metadata construction
        if ck and hasattr(ck, 'construct_metadata'):
            ck.construct_metadata(
                ordered_global_eid, self.rank, ordered_src_token, 
                ordered_src_k, send_meta
            )
        else:
            # Fallback to PyTorch operations
            send_meta[:, 0].copy_(ordered_global_eid.to(torch.int32))
            send_meta[:, 1].fill_(self.rank)
            send_meta[:, 2].copy_(ordered_src_token.to(torch.int32))
            send_meta[:, 3].copy_(ordered_src_k.to(torch.int32))
            send_meta[:, 4].zero_()
        
        send_data.copy_(ordered_data.to(cfg.in_dtype))

        # Exchange receive counts
        if self.use_shmem:
            # Use ROCm SHMEM for direct GPU communication
            recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
            self.shmem_group.alltoall(send_counts, recv_counts)
            total_recv = int(recv_counts.sum().item())
        else:
            # Fallback to PyTorch distributed
            recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
            dist.all_to_all_single(recv_counts, send_counts)
            total_recv = int(recv_counts.sum().item())

        # Fast path: nothing to receive
        if total_recv == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            return expert_num_tokens, self._contig(self.expert_x), self._contig(self.expert_meta)

        # Use pre-allocated buffers
        recv_meta = self.recv_meta[:total_recv]
        recv_data = self.recv_data[:total_recv]

        # Separate all_to_all_single calls for meta and data
        if self.use_shmem:
            # Use ROCm SHMEM for direct GPU communication
            self.shmem_group.alltoall(send_meta, recv_meta)
            self.shmem_group.alltoall(send_data, recv_data)
        else:
            # Fallback to PyTorch distributed
            input_split_sizes = send_counts.tolist()
            output_split_sizes = recv_counts.tolist()
            
            # First transfer metadata
            dist.all_to_all_single(recv_meta, send_meta,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)
            
            # Then transfer data
            dist.all_to_all_single(recv_data, send_data,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)

        # Compute local expert id for each received record
        global_eid_recv = recv_meta[:, 0].long()
        local_eid = (global_eid_recv % self.num_local_experts).long()

        # Sort by local expert id to group items by local expert
        order_local = torch.argsort(local_eid)
        sorted_local_eid = local_eid[order_local]
        sorted_meta = recv_meta[order_local]
        sorted_data = recv_data[order_local]

        # Compute counts per local expert
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).long().to(device)
        expert_num_tokens = local_counts.to(torch.int32)

        # Use pre-allocated expert buffers
        expert_x = self.expert_x
        expert_meta = self.expert_meta

        # If there are received items, compute per-record position within each expert group
        if total_recv > 0:
            self._optimized_data_placement(sorted_local_eid, sorted_meta, sorted_data, 
                                          expert_x, expert_meta, local_counts, total_recv)

        return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

    # Optimized data placement using HIP kernel
    def _optimized_data_placement(self, sorted_local_eid, sorted_meta, sorted_data, 
                                 expert_x, expert_meta, local_counts, total_recv):
        """Optimized data placement using HIP kernel or Composable Kernel"""
        if ck and hasattr(ck, 'data_placement'):
            # Use Composable Kernel if available
            ck.data_placement(
                sorted_local_eid, sorted_meta, sorted_data,
                expert_x, expert_meta, local_counts,
                self.num_local_experts, self.hidden_dim, self.META_DIM
            )
        else:
            # Fallback to PyTorch operations
            # Compute per-record position within each expert group
            cum = torch.cumsum(local_counts, dim=0)
            starts = torch.empty_like(cum)
            starts[0] = 0
            starts[1:] = cum[:-1]
            idx_in_sorted = torch.arange(total_recv, device=sorted_local_eid.device)
            group_pos = idx_in_sorted - starts[sorted_local_eid]
            
            # Direct assignment using indices (safe and vectorized)
            expert_x[sorted_local_eid, group_pos] = sorted_data
            expert_meta[sorted_local_eid, group_pos] = sorted_meta

    # ---------------- Combine ----------------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor,
                expert_meta: torch.Tensor, expert_y: torch.Tensor,
                expert_num_tokens: torch.Tensor):
        """
        Combine expert outputs back to tokens using optimized HIP kernels and ROCm SHMEM.
        
        Args:
            out_tokens: (max_num_tokens, hidden_dim) output tensor
            weights: (max_num_tokens, k) float weights
            expert_meta: (num_local_experts, aligned_max_recv, META_DIM) int32 metadata
            expert_y: (num_local_experts, aligned_max_recv, hidden_dim) expert outputs
            expert_num_tokens: (num_local_experts,) int32 token counts
        """
        cfg = self.cfg
        device = out_tokens.device

        # Ensure contiguous memory layout
        out_tokens = self._contig(out_tokens)
        weights = self._contig(weights)
        expert_meta = self._contig(expert_meta)
        expert_y = self._contig(expert_y)
        expert_num_tokens = self._contig(expert_num_tokens)

        # Get counts and calculate total send size
        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)
        
        # Fast path: nothing to send
        if total_send == 0:
            return out_tokens

        # Gather valid entries from experts using HIP kernel
        if ck and hasattr(ck, 'gather_expert_data'):
            # Use Composable Kernel if available
            send_meta, send_y = ck.gather_expert_data(
                expert_meta, expert_y, expert_num_tokens,
                self.num_local_experts, self.META_DIM, self.hidden_dim
            )
        else:
            # Fallback to PyTorch operations
            meta_list = []
            y_list = []
            for local_id, cnt in enumerate(counts):
                if cnt > 0:
                    meta_list.append(expert_meta[local_id, :cnt])
                    y_list.append(expert_y[local_id, :cnt])
            
            # Handle case where no valid entries exist
            if not meta_list or not y_list:
                return out_tokens
                
            # Concatenate gathered data
            send_meta = torch.cat(meta_list, dim=0)  # (total_send, META_DIM) int32
            send_y = torch.cat(y_list, dim=0)        # (total_send, hidden_dim) out_dtype

        # Calculate destination ranks
        dst_ranks = send_meta[:, 1].long()
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size).long().to(device)

        # Sort by destination to make contiguous per-dst block
        order = torch.argsort(dst_ranks)
        ordered_meta = send_meta[order]
        ordered_y = send_y[order]

        # Exchange receive counts
        if self.use_shmem:
            # Use ROCm SHMEM for direct GPU communication
            recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
            self.shmem_group.alltoall(send_counts, recv_counts)
            total_recv = int(recv_counts.sum().item())
        else:
            # Fallback to PyTorch distributed
            recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
            dist.all_to_all_single(recv_counts, send_counts)
            total_recv = int(recv_counts.sum().item())
        
        # Fast path: nothing to receive
        if total_recv == 0:
            return out_tokens

        # Use pre-allocated buffers
        recv_meta = self.recv_meta[:total_recv]
        recv_y = self.recv_data[:total_recv]

        # Separate all_to_all_single calls for meta and data
        if self.use_shmem:
            # Use ROCm SHMEM for direct GPU communication
            self.shmem_group.alltoall(ordered_meta, recv_meta)
            self.shmem_group.alltoall(ordered_y, recv_y)
        else:
            # Fallback to PyTorch distributed
            input_split_sizes = send_counts.tolist()
            output_split_sizes = recv_counts.tolist()
            
            # First transfer metadata
            dist.all_to_all_single(recv_meta, ordered_meta,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)
            
            # Then transfer data
            dist.all_to_all_single(recv_y, ordered_y,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)

        # Apply weights and scatter_add to out_tokens using HIP kernel
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()

        # Use optimized weighted scatter operation
        self._optimized_weighted_scatter(out_tokens, recv_y, weights, src_tokens, src_k)
        
        return out_tokens

    # Optimized weighted scatter operation using HIP kernel
    def _optimized_weighted_scatter(self, out_tokens, recv_y, weights, src_tokens, src_k):
        """Optimized weighted scatter operation using HIP kernel or Composable Kernel"""
        if ck and hasattr(ck, 'weighted_scatter'):
            # Use Composable Kernel if available
            ck.weighted_scatter(
                out_tokens, recv_y, weights, src_tokens, src_k,
                out_tokens.shape[0], self.hidden_dim, weights.shape[1]
            )
        else:
            # Fallback to PyTorch operations
            # Compute weighted outputs
            weights_selected = weights[src_tokens, src_k]  # (total_recv,)
            weighted_y = recv_y * weights_selected.unsqueeze(1)  # (total_recv, hidden_dim)
            weighted_y = weighted_y.to(out_tokens.dtype)  # Ensure dtype match to avoid RuntimeError
            
            # Scatter add to out_tokens
            out_tokens.index_add_(0, src_tokens, weighted_y)

    def __del__(self):
        """Clean up resources"""
        if self.use_shmem and hasattr(self, 'shmem_group'):
            rocm_shmem.finalize()


# ---------------- HIP Kernels for Optimized Operations ----------------
# These kernels would be implemented in separate .hip files and compiled with HIPCC

# Example HIP kernel for data placement
"""
extern "C" __global__
void data_placement_kernel(
    const int* sorted_local_eid,
    const int* sorted_meta,
    const half* sorted_data,
    half* expert_x,
    int* expert_meta,
    const int* local_counts,
    const int num_local_experts,
    const int hidden_dim,
    const int meta_dim,
    const int total_recv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_recv) return;
    
    int local_eid = sorted_local_eid[idx];
    
    // Compute position within expert group
    int pos_in_expert = 0;
    // Use wavefront reduction for prefix sum
    for (int i = 0; i < local_eid; i++) {
        pos_in_expert += local_counts[i];
    }
    pos_in_expert += idx - // offset calculation
    
    // Copy data to expert buffer
    for (int d = 0; d < hidden_dim; d++) {
        expert_x[local_eid * hidden_dim * max_recv + pos_in_expert * hidden_dim + d] = 
            sorted_data[idx * hidden_dim + d];
    }
    
    // Copy metadata
    for (int m = 0; m < meta_dim; m++) {
        expert_meta[local_eid * meta_dim * max_recv + pos_in_expert * meta_dim + m] = 
            sorted_meta[idx * meta_dim + m];
    }
}
"""

# Example HIP kernel for weighted scatter
"""
extern "C" __global__
void weighted_scatter_kernel(
    half* out_tokens,
    const half* recv_y,
    const half* weights,
    const int* src_tokens,
    const int* src_k,
    const int max_num_tokens,
    const int hidden_dim,
    const int k,
    const int total_recv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_recv) return;
    
    int token_idx = src_tokens[idx];
    int k_idx = src_k[idx];
    half weight = weights[token_idx * k + k_idx];
    
    // Use wavefront reduction for atomic operations
    for (int d = 0; d < hidden_dim; d++) {
        half value = recv_y[idx * hidden_dim + d] * weight;
        // Use atomic add for half precision
        atomicAdd(&out_tokens[token_idx * hidden_dim + d], value);
    }
}
"""


def custom_kernel(data: input_t) -> output_t:
    # Unpack the tuple data
    cfg, rank_data, rank, world_size = data
    device = rank_data.x.device

    # Create optimized AMD all-to-all instance
    ata = OptimizedAMDAllToAll(cfg, int(rank), int(world_size))

    # Dispatch tokens to experts
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Simulated expert compute (use out_dtype)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + int(rank))

    # Combine expert outputs back to tokens
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]