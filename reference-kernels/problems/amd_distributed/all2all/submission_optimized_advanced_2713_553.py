import torch
import torch.distributed as dist
from task import input_t, output_t


# ---------------- Advanced AMD ROCm Optimized All2All Implementation ----------------
class AdvancedAMDOptimizedAllToAll:
    META_DIM = 5  # [global_exp, src_rank, src_token, src_k, pad]

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden_dim = cfg.hidden_dim
        # AMD-specific optimization: align with wavefront size (64 work-items)
        self.wavefront_size = 64
        # Additional AMD optimizations for MI300 series
        self.alignment_factor = 128  # For better memory bandwidth utilization
        self.cache_line_size = 64    # AMD cache line size in bytes

    def _align_dimension(self, size: int, alignment: int = None) -> int:
        """Align dimension to specified alignment factor."""
        if alignment is None:
            alignment = self.wavefront_size
        return ((size + alignment - 1) // alignment) * alignment

    def _ensure_contiguous(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is contiguous for optimal memory access."""
        return tensor.contiguous() if not tensor.is_contiguous() else tensor

    # ---------- dispatch: optimized with packed data transfer ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Dispatch tokens to experts using optimized packed all-to-all for AMD ROCm.
        """
        cfg = self.cfg
        device = dp_x.device
        num_tokens, k = indices.shape

        # Ensure input tensors are contiguous
        dp_x = self._ensure_contiguous(dp_x)
        indices = self._ensure_contiguous(indices)

        # Compute destination information using vectorized operations
        # This avoids Python loops and leverages PyTorch's optimized operations
        dst_ranks = torch.div(indices, self.num_local_experts, rounding_mode='floor')
        dst_flat = dst_ranks.reshape(-1)
        
        # Create source token indices efficiently
        src_token_idx = torch.arange(num_tokens, device=device).repeat_interleave(k)
        global_eids_flat = indices.reshape(-1)

        # Calculate send counts per rank using bincount (vectorized)
        send_counts = torch.bincount(dst_flat, minlength=self.world_size)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        
        total_send = send_counts.sum().item()
        total_recv = recv_counts.sum().item()

        per_row_len = self.META_DIM + self.hidden_dim

        # Prepare send buffer with AMD optimizations
        if total_send == 0:
            combined_send = torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        else:
            # Sort by destination for contiguous memory access
            order = torch.argsort(dst_flat)
            ordered_token_idx = src_token_idx[order]
            ordered_global_eids = global_eids_flat[order]
            
            # Gather data in order using advanced indexing
            send_data = dp_x[ordered_token_idx]
            
            # Create metadata using vectorized operations
            k_idx = torch.arange(k, device=device).repeat(num_tokens)[order]
            send_meta = torch.stack([
                ordered_global_eids.int(),
                torch.full_like(ordered_global_eids, self.rank, dtype=torch.int32),
                ordered_token_idx.int(),
                k_idx.int(),
                torch.zeros_like(ordered_global_eids, dtype=torch.int32)
            ], dim=1)

            # Pack metadata and data as float32 for efficient transfer
            # This reduces communication overhead by using a single all_to_all_single call
            send_meta_float = send_meta.float()
            send_data_float = send_data.float()
            
            # Ensure packed data is contiguous
            send_meta_float = self._ensure_contiguous(send_meta_float)
            send_data_float = self._ensure_contiguous(send_data_float)
            
            combined_send = torch.cat([send_meta_float, send_data_float], dim=1)
            combined_send = self._ensure_contiguous(combined_send)

        # Prepare receive buffer with optimal alignment for AMD GPU memory access
        aligned_total_recv = self._align_dimension(total_recv, self.alignment_factor)
        recv_combined = torch.empty((aligned_total_recv, per_row_len), dtype=torch.float32, device=device)
        recv_combined = self._ensure_contiguous(recv_combined)

        # Single all_to_all_single for packed data - this is a key optimization
        # that reduces communication overhead compared to separate metadata and data transfers
        if total_send > 0 and total_recv > 0:
            dist.all_to_all_single(
                recv_combined[:total_recv],  # Use only the needed portion
                combined_send,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts_t.tolist()
            )

        # Handle empty receive case
        if total_recv == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            expert_x = torch.empty((self.num_local_experts, self.max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            # Ensure contiguous memory layout
            expert_x = self._ensure_contiguous(expert_x)
            expert_meta = self._ensure_contiguous(expert_meta)
            return expert_num_tokens, expert_x, expert_meta

        # Unpack received data efficiently
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

        # Count tokens per expert using bincount (vectorized)
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).int()
        expert_num_tokens = local_counts.clone()

        # Allocate output tensors with optimal alignment for AMD GPUs
        aligned_max_recv = self._align_dimension(self.max_recv, self.alignment_factor)
        expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim), 
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM), 
                                  dtype=torch.int32, device=device)
        
        # Ensure contiguous memory layout for optimal performance
        expert_x = self._ensure_contiguous(expert_x)
        expert_meta = self._ensure_contiguous(expert_meta)

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

    # ---------- combine: optimized packed all_to_all back ----------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, 
                expert_meta: torch.Tensor, expert_y: torch.Tensor, 
                expert_num_tokens: torch.Tensor):
        """
        Combine expert outputs back to tokens using optimized packed all-to-all for AMD ROCm.
        """
        cfg = self.cfg
        device = out_tokens.device

        # Ensure input tensors are contiguous for optimal memory access
        out_tokens = self._ensure_contiguous(out_tokens)
        weights = self._ensure_contiguous(weights)
        expert_meta = self._ensure_contiguous(expert_meta)
        expert_y = self._ensure_contiguous(expert_y)
        expert_num_tokens = self._ensure_contiguous(expert_num_tokens)

        # Get counts and calculate total send size
        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)

        # Gather send entries (y + meta) using vectorized operations
        if total_send == 0:
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
                # Ensure concatenated tensors are contiguous
                send_meta = self._ensure_contiguous(send_meta)
                send_y = self._ensure_contiguous(send_y)
            else:
                send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
                send_y = torch.empty((0, self.hidden_dim), dtype=cfg.out_dtype, device=device)

        # Calculate destination ranks using vectorized operations
        dst_ranks = send_meta[:, 1].long() if total_send > 0 else torch.empty((0,), dtype=torch.long, device=device)
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size) if total_send > 0 else torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        total_recv = recv_counts.sum().item()

        per_row_len = self.META_DIM + self.hidden_dim

        # Pack send data with AMD optimizations
        if total_send == 0:
            combined_send = torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        else:
            # Sort by destination rank for better memory access patterns
            order = torch.argsort(dst_ranks)
            ordered_meta = send_meta[order]
            ordered_y = send_y[order]
            
            # Pack as float32 for efficient transfer
            ordered_meta_float = ordered_meta.float()
            ordered_y_float = ordered_y.float()
            
            # Ensure packed data is contiguous
            ordered_meta_float = self._ensure_contiguous(ordered_meta_float)
            ordered_y_float = self._ensure_contiguous(ordered_y_float)
            
            combined_send = torch.cat([ordered_meta_float, ordered_y_float], dim=1)
            combined_send = self._ensure_contiguous(combined_send)

        # Prepare receive buffer with optimal alignment
        aligned_total_recv = self._align_dimension(total_recv, self.alignment_factor)
        recv_combined = torch.empty((aligned_total_recv, per_row_len), dtype=torch.float32, device=device)
        recv_combined = self._ensure_contiguous(recv_combined)
        
        # Single all_to_all_single for packed data - key optimization
        if total_send > 0 and total_recv > 0:
            dist.all_to_all_single(
                recv_combined[:total_recv],  # Use only the needed portion
                combined_send,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts_t.tolist()
            )

        # Handle empty receive case
        if total_recv == 0:
            return out_tokens

        # Unpack received data efficiently
        recv_meta_float = recv_combined[:total_recv, :self.META_DIM]
        recv_data_float = recv_combined[:total_recv, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.out_dtype)

        # Extract metadata using vectorized operations
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()
        
        # Apply weights using vectorized operations for better performance
        w = weights[src_tokens, src_k].float().unsqueeze(1)
        recv_buf_float = recv_buf.float()
        weighted = recv_buf_float * w

        # Vectorized scatter_add with AMD optimizations
        # This is more efficient than looping through each token
        if out_tokens.dtype != torch.float32:
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
    
    # Create advanced AMD optimized all-to-all instance
    ata = AdvancedAMDOptimizedAllToAll(cfg, rank, world_size)

    # Dispatch tokens to experts
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Expert computation (simulated)
    # AMD optimization: Use the configured output dtype for better performance on AMD GPUs
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # Combine expert outputs back to tokens
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]