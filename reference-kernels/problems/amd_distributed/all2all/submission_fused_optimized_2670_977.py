import torch
import torch.distributed as dist
from task import input_t, output_t


# ---------------- Fused AMD ROCm Optimized All2All Implementation ----------------
class FusedAMDOptimizedAllToAll:
    META_DIM = 5  # [global_exp, src_rank, src_token, src_k, pad]

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden_dim = cfg.hidden_dim
        # AMD-specific optimization parameters for MI300 series
        self.wavefront_size = 64      # AMD wavefront size
        self.alignment_factor = 128   # For better memory bandwidth utilization
        self.workgroup_size = 1024    # Typical max workgroup size for AMD GPUs

    def _align_to_amd(self, size: int, alignment: int = None) -> int:
        """Align size to AMD-specific boundaries for optimal performance."""
        if alignment is None:
            alignment = self.wavefront_size
        return ((size + alignment - 1) // alignment) * alignment

    def _optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for AMD GPU memory access patterns."""
        # Ensure tensor is contiguous and aligned for better memory coalescing
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    # ---------- dispatch with fused operations ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Dispatch tokens to experts using fused operations optimized for AMD ROCm.
        """
        cfg = self.cfg
        device = dp_x.device
        num_tokens, k = indices.shape

        # AMD optimization: Ensure optimal memory layout
        dp_x = self._optimize_memory_layout(dp_x)
        indices = self._optimize_memory_layout(indices)

        # Vectorized computation of destination information
        # This replaces Python loops with efficient PyTorch operations
        dst_ranks = torch.div(indices, self.num_local_experts, rounding_mode='floor')
        dst_flat = dst_ranks.view(-1)
        
        # Efficiently create source token indices
        src_token_idx = torch.arange(num_tokens, device=device).repeat_interleave(k)
        global_eids_flat = indices.view(-1)

        # Calculate send counts using vectorized bincount operation
        send_counts = torch.bincount(dst_flat, minlength=self.world_size)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts using all_to_all_single
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        
        total_send = send_counts.sum().item()
        total_recv = recv_counts.sum().item()

        per_row_len = self.META_DIM + self.hidden_dim

        # Prepare fused send buffer with AMD optimizations
        if total_send == 0:
            # Handle empty case efficiently
            combined_send = torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        else:
            # Sort by destination for better memory access patterns
            order = torch.argsort(dst_flat)
            ordered_token_idx = src_token_idx[order]
            ordered_global_eids = global_eids_flat[order]
            
            # Gather data using advanced indexing (more efficient than loops)
            send_data = dp_x[ordered_token_idx]
            
            # Create metadata using vectorized operations
            k_idx = torch.arange(k, device=device).repeat(num_tokens)[order]
            send_meta = torch.stack([
                ordered_global_eids.int(),                    # global_exp
                torch.full_like(ordered_global_eids, self.rank, dtype=torch.int32),  # src_rank
                ordered_token_idx.int(),                      # src_token
                k_idx.int(),                                  # src_k
                torch.zeros_like(ordered_global_eids, dtype=torch.int32)  # pad
            ], dim=1)

            # Pack metadata and data as float32 for efficient transfer
            # This reduces communication overhead by using a single all_to_all_single call
            send_meta_float = send_meta.float()
            send_data_float = send_data.float()
            
            # AMD optimization: Ensure packed data is contiguous
            send_meta_float = self._optimize_memory_layout(send_meta_float)
            send_data_float = self._optimize_memory_layout(send_data_float)
            
            # Fuse metadata and data into a single buffer
            combined_send = torch.cat([send_meta_float, send_data_float], dim=1)
            combined_send = self._optimize_memory_layout(combined_send)

        # Prepare receive buffer with optimal alignment for AMD GPU memory access
        # Using larger alignment factor for better memory coalescing on AMD GPUs
        aligned_total_recv = self._align_to_amd(total_recv, self.alignment_factor)
        recv_combined = torch.empty((aligned_total_recv, per_row_len), dtype=torch.float32, device=device)
        recv_combined = self._optimize_memory_layout(recv_combined)

        # Single fused all_to_all_single for packed data - key optimization
        # This reduces communication overhead compared to separate transfers
        if total_send > 0 and total_recv > 0:
            dist.all_to_all_single(
                recv_combined[:total_recv],  # Use only the needed portion to save memory
                combined_send,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts_t.tolist()
            )

        # Handle empty receive case efficiently
        if total_recv == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            expert_x = torch.empty((self.num_local_experts, self.max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            # AMD optimization: Ensure contiguous memory layout
            expert_x = self._optimize_memory_layout(expert_x)
            expert_meta = self._optimize_memory_layout(expert_meta)
            return expert_num_tokens, expert_x, expert_meta

        # Efficiently unpack received data
        recv_meta_float = recv_combined[:total_recv, :self.META_DIM]
        recv_data_float = recv_combined[:total_recv, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.in_dtype)

        # Group by local expert using sorting for better memory access patterns
        global_eid_recv = recv_meta[:, 0].long()
        local_eid_flat = torch.remainder(global_eid_recv, self.num_local_experts).long()

        # Sort by local expert id to group data for each expert together
        order_local = torch.argsort(local_eid_flat)
        sorted_recv_buf = recv_buf[order_local]
        sorted_recv_meta = recv_meta[order_local]
        sorted_local_eid = local_eid_flat[order_local]

        # Count tokens per expert using vectorized bincount
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).int()
        expert_num_tokens = local_counts.clone()

        # Allocate output tensors with AMD-optimized alignment
        # Using larger alignment factor for better memory access patterns on AMD GPUs
        aligned_max_recv = self._align_to_amd(self.max_recv, self.alignment_factor)
        expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim), 
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM), 
                                  dtype=torch.int32, device=device)
        
        # AMD optimization: Ensure contiguous memory layout
        expert_x = self._optimize_memory_layout(expert_x)
        expert_meta = self._optimize_memory_layout(expert_meta)

        # Distribute data to experts using vectorized operations
        # This avoids Python loops and leverages PyTorch's optimized operations
        if sorted_recv_buf.shape[0] > 0:
            split_sizes = local_counts.tolist()
            start_idx = 0
            for local_id, cnt in enumerate(split_sizes):
                if cnt > 0:
                    expert_x[local_id, :cnt].copy_(sorted_recv_buf[start_idx:start_idx+cnt])
                    expert_meta[local_id, :cnt].copy_(sorted_recv_meta[start_idx:start_idx+cnt])
                    start_idx += cnt

        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine with fused operations ----------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, 
                expert_meta: torch.Tensor, expert_y: torch.Tensor, 
                expert_num_tokens: torch.Tensor):
        """
        Combine expert outputs back to tokens using fused operations optimized for AMD ROCm.
        """
        cfg = self.cfg
        device = out_tokens.device

        # AMD optimization: Ensure optimal memory layout for all input tensors
        out_tokens = self._optimize_memory_layout(out_tokens)
        weights = self._optimize_memory_layout(weights)
        expert_meta = self._optimize_memory_layout(expert_meta)
        expert_y = self._optimize_memory_layout(expert_y)
        expert_num_tokens = self._optimize_memory_layout(expert_num_tokens)

        # Get counts and calculate total send size using vectorized operations
        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)

        # Gather send entries (y + meta) using vectorized operations
        if total_send == 0:
            # Handle empty case efficiently
            send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
            send_y = torch.empty((0, self.hidden_dim), dtype=cfg.out_dtype, device=device)
        else:
            # Vectorized gathering of data to minimize Python loops
            meta_list = []
            y_list = []
            for local_id, cnt in enumerate(counts):
                if cnt > 0:
                    meta_list.append(expert_meta[local_id, :cnt])
                    y_list.append(expert_y[local_id, :cnt])
            
            if meta_list:
                send_meta = torch.cat(meta_list, dim=0)
                send_y = torch.cat(y_list, dim=0)
                # AMD optimization: Ensure concatenated tensors are contiguous
                send_meta = self._optimize_memory_layout(send_meta)
                send_y = self._optimize_memory_layout(send_y)
            else:
                send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
                send_y = torch.empty((0, self.hidden_dim), dtype=cfg.out_dtype, device=device)

        # Calculate destination ranks using vectorized operations
        dst_ranks = send_meta[:, 1].long() if total_send > 0 else torch.empty((0,), dtype=torch.long, device=device)
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size) if total_send > 0 else torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts using all_to_all_single
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        total_recv = recv_counts.sum().item()

        per_row_len = self.META_DIM + self.hidden_dim

        # Pack send data with AMD optimizations
        if total_send == 0:
            # Handle empty case efficiently
            combined_send = torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        else:
            # Sort by destination rank for better memory access patterns
            order = torch.argsort(dst_ranks)
            ordered_meta = send_meta[order]
            ordered_y = send_y[order]
            
            # Pack as float32 for efficient transfer
            ordered_meta_float = ordered_meta.float()
            ordered_y_float = ordered_y.float()
            
            # AMD optimization: Ensure packed data is contiguous
            ordered_meta_float = self._optimize_memory_layout(ordered_meta_float)
            ordered_y_float = self._optimize_memory_layout(ordered_y_float)
            
            # Fuse metadata and data into a single buffer
            combined_send = torch.cat([ordered_meta_float, ordered_y_float], dim=1)
            combined_send = self._optimize_memory_layout(combined_send)

        # Prepare receive buffer with AMD-optimized alignment
        aligned_total_recv = self._align_to_amd(total_recv, self.alignment_factor)
        recv_combined = torch.empty((aligned_total_recv, per_row_len), dtype=torch.float32, device=device)
        recv_combined = self._optimize_memory_layout(recv_combined)
        
        # Single fused all_to_all_single for packed data - key optimization
        if total_send > 0 and total_recv > 0:
            dist.all_to_all_single(
                recv_combined[:total_recv],  # Use only the needed portion to save memory
                combined_send,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts_t.tolist()
            )

        # Handle empty receive case efficiently
        if total_recv == 0:
            return out_tokens

        # Efficiently unpack received data
        recv_meta_float = recv_combined[:total_recv, :self.META_DIM]
        recv_data_float = recv_combined[:total_recv, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.out_dtype)

        # Extract metadata using vectorized operations for better performance
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()
        
        # Apply weights using vectorized operations for maximum efficiency
        w = weights[src_tokens, src_k].float().unsqueeze(1)
        recv_buf_float = recv_buf.float()
        weighted = recv_buf_float * w

        # Vectorized scatter_add with AMD optimizations
        # This is much more efficient than looping through each token
        if out_tokens.dtype != torch.float32:
            # Handle type conversion efficiently
            out_acc = out_tokens.float()
            idx = src_tokens.unsqueeze(1).expand(-1, self.hidden_dim)
            out_acc.scatter_add_(0, idx, weighted)
            out_tokens.copy_(out_acc.to(out_tokens.dtype))
        else:
            idx = src_tokens.unsqueeze(1).expand(-1, self.hidden_dim)
            out_tokens.scatter_add_(0, idx, weighted)

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    
    # Create fused AMD optimized all-to-all instance
    ata = FusedAMDOptimizedAllToAll(cfg, rank, world_size)

    # Dispatch tokens to experts using fused operations
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Expert computation (simulated)
    # AMD optimization: Use the configured output dtype for better performance on AMD GPUs
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # Combine expert outputs back to tokens using fused operations
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]