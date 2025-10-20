import torch
import torch.distributed as dist
from task import input_t, output_t
import triton
import triton.language as tl


# ---------------- Fixed High Performance AMD AllToAll Implementation ----------------
class FixedHighPerformanceAMDAllToAll:
    """
    Fixed high performance implementation optimized for AMD ROCm with proper synchronization:
    - Separates meta (int32) and data (floatX) transmission for reduced overhead
    - Uses AMD wavefront-aware memory alignment (64-byte boundaries)
    - Employs vectorized operations instead of Python loops
    - Minimizes temporary memory allocations and dtype conversions
    - Uses advanced indexing for efficient data placement
    - Implements proper synchronization to avoid deadlocks
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
        Compute flattened destination information for tokens.
        
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
        
        # Compute destination ranks for each token-expert pair
        dst = torch.div(indices, self.num_local_experts, rounding_mode='floor').reshape(-1).long()
        
        # Create source token indices
        src_token = torch.arange(num_tokens, device=device).repeat_interleave(k).long()
        global_eid = indices.reshape(-1).long()
        src_k = torch.arange(k, device=device).repeat(num_tokens).long()
        
        return dst, src_token, global_eid, src_k

    # ---------------- Dispatch ----------------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Dispatch tokens to experts using optimized separated transmission.
        
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
            aligned_max_recv = self._align(self.max_recv, self.alignment)
            expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

        # Sort by destination to create contiguous send buffer per rank
        order = torch.argsort(dst_flat)
        ordered_dst = dst_flat[order]
        ordered_src_token = src_token_idx[order]
        ordered_global_eid = global_eids_flat[order]
        ordered_src_k = src_k_flat[order]
        ordered_data = dp_x[ordered_src_token]

        # Construct send metadata (int32) and send data (in_dtype)
        send_meta = torch.stack([
            ordered_global_eid.to(torch.int32),
            torch.full_like(ordered_global_eid, fill_value=self.rank, dtype=torch.int32),
            ordered_src_token.to(torch.int32),
            ordered_src_k.to(torch.int32),
            torch.zeros_like(ordered_global_eid, dtype=torch.int32)
        ], dim=1)
        send_meta = self._contig(send_meta)
        send_data = self._contig(ordered_data.to(cfg.in_dtype))

        # Exchange receive counts
        recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts, send_counts)
        total_recv = int(recv_counts.sum().item())

        # Align receive buffer size for better memory access patterns
        aligned_recv = self._align(total_recv, self.alignment) if total_recv > 0 else self.alignment

        # Allocate receive buffers
        recv_meta = torch.empty((aligned_recv, self.META_DIM), dtype=torch.int32, device=device)
        recv_data = torch.empty((aligned_recv, self.hidden_dim), dtype=cfg.in_dtype, device=device)

        # Separate all_to_all_single calls for meta and data with proper synchronization
        if total_send > 0 and total_recv > 0:
            # Ensure split sizes are correctly sized
            input_split_sizes = send_counts.tolist()
            output_split_sizes = recv_counts.tolist()
            
            # First transfer metadata
            dist.all_to_all_single(recv_meta[:total_recv], send_meta,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)
            
            # Then transfer data
            dist.all_to_all_single(recv_data[:total_recv], send_data,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)

        # Handle empty receive case
        if total_recv == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            aligned_max_recv = self._align(self.max_recv, self.alignment)
            expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

        # Trim buffers to actual received size
        recv_meta = recv_meta[:total_recv]
        recv_data = recv_data[:total_recv]

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

        # Allocate aligned expert buffers with AMD-specific alignment
        aligned_max_recv = self._align(self.max_recv, self.alignment)
        expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                  dtype=torch.int32, device=device)

        # If there are received items, compute per-record position within each expert group
        if total_recv > 0:
            self._optimized_data_placement(sorted_local_eid, sorted_meta, sorted_data, 
                                          expert_x, expert_meta, local_counts, total_recv)

        return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

    # Optimized data placement using PyTorch (fallback from Triton to avoid memory faults)
    def _optimized_data_placement(self, sorted_local_eid, sorted_meta, sorted_data, 
                                 expert_x, expert_meta, local_counts, total_recv):
        """Optimized data placement using PyTorch scatter for stability"""
        # Compute per-record position within each expert group (already calculated earlier)
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
        Combine expert outputs back to tokens using optimized separated transmission.
        
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

        # Gather valid entries from experts using vectorized operations
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
        recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts, send_counts)
        total_recv = int(recv_counts.sum().item())
        
        # Fast path: nothing to receive
        if total_recv == 0:
            return out_tokens

        # Align receive buffer size
        aligned_recv = self._align(total_recv, self.alignment) if total_recv > 0 else self.alignment
        recv_meta = torch.empty((aligned_recv, self.META_DIM), dtype=torch.int32, device=device)
        recv_y = torch.empty((aligned_recv, self.hidden_dim), dtype=cfg.out_dtype, device=device)

        # Separate all_to_all_single calls for meta and data with proper synchronization
        if total_send > 0 and total_recv > 0:
            # Ensure split sizes are correctly sized
            input_split_sizes = send_counts.tolist()
            output_split_sizes = recv_counts.tolist()
            
            # First transfer metadata
            dist.all_to_all_single(recv_meta[:total_recv], ordered_meta,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)
            
            # Then transfer data
            dist.all_to_all_single(recv_y[:total_recv], ordered_y,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes)

        # Trim buffers to actual received size
        recv_meta = recv_meta[:total_recv]
        recv_y = recv_y[:total_recv]

        # Apply weights and scatter_add to out_tokens using vectorized operations
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()

        # Use optimized weighted scatter operation
        self._optimized_weighted_scatter(out_tokens, recv_y, weights, src_tokens, src_k)
        
        return out_tokens

    # Optimized weighted scatter operation using PyTorch (fallback from Triton to avoid memory faults)
    def _optimized_weighted_scatter(self, out_tokens, recv_y, weights, src_tokens, src_k):
        """Optimized weighted scatter operation using PyTorch index_add for stability"""
        # Compute weighted outputs
        weights_selected = weights[src_tokens, src_k]  # (total_recv,)
        weighted_y = recv_y * weights_selected.unsqueeze(1)  # (total_recv, hidden_dim)
        weighted_y = weighted_y.to(out_tokens.dtype)  # Ensure dtype match to avoid RuntimeError
        
        # Scatter add to out_tokens
        out_tokens.index_add_(0, src_tokens, weighted_y)


# ---------------- Triton Kernels for Optimized Operations ----------------
# Note: Triton kernels are disabled in this version to avoid memory access faults.
# They can be re-enabled after verifying PyTorch fallback correctness.

@triton.jit
def _optimized_placement_kernel(
    sorted_data_ptr, sorted_meta_ptr, expert_x_ptr, expert_meta_ptr,
    sorted_local_eid_ptr, group_pos_ptr, total_recv,
    expert_data_stride, expert_meta_stride,
    hidden_dim: tl.constexpr, meta_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # This kernel is disabled; use PyTorch fallback instead
    pass


@triton.jit
def _optimized_scatter_kernel(
    out_tokens_ptr, recv_y_ptr, weights_ptr, src_tokens_ptr, src_k_ptr,
    total_recv, hidden_dim, k,
    BLOCK_SIZE: tl.constexpr
):
    # This kernel is disabled; use PyTorch fallback instead
    pass


def custom_kernel(data: input_t) -> output_t:
    # Unpack the tuple data
    cfg, rank_data, rank, world_size = data
    device = rank_data.x.device

    # Create fixed high performance AMD all-to-all instance
    ata = FixedHighPerformanceAMDAllToAll(cfg, int(rank), int(world_size))

    # Dispatch tokens to experts
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Simulated expert compute (use out_dtype)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + int(rank))

    # Combine expert outputs back to tokens
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]