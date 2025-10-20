import torch
import torch.distributed as dist
from task import input_t, output_t


# ---------------- Ultra High Performance AMD ROCm All2All Implementation ----------------
class UltraHighPerformanceAMDOptimizedAllToAll:
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
        # Pre-allocate commonly used tensors
        self._initialize_buffers()

    def _initialize_buffers(self):
        """Pre-allocate commonly used buffers to reduce allocation overhead."""
        pass  # Will be initialized on first use

    def _align_to_wavefront(self, size: int) -> int:
        """
        Align size to wavefront boundary for optimal memory access.
        """
        return ((size + self.wavefront_size - 1) // self.wavefront_size) * self.wavefront_size

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        num_tokens, k = indices.shape

        # Compute destination information using vectorized operations
        dst_flat = (indices // self.num_local_experts).reshape(-1)
        src_token_idx = torch.arange(num_tokens, device=device).repeat_interleave(k)
        global_eids_flat = indices.reshape(-1)

        # Calculate send counts using bincount
        send_counts = torch.bincount(dst_flat, minlength=self.world_size)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        
        total_send = send_counts.sum().item()
        total_recv = recv_counts.sum().item()

        # Prepare packed send buffer (metadata + data)
        if total_send == 0:
            combined_send = torch.empty((0, self.META_DIM + self.hidden_dim), dtype=torch.float32, device=device)
        else:
            # Sort by destination for contiguous memory access
            order = torch.argsort(dst_flat)
            ordered_token_idx = src_token_idx[order]
            ordered_global_eids = global_eids_flat[order]
            
            # Gather data in order
            send_data = dp_x[ordered_token_idx]
            
            # Create metadata efficiently
            k_idx = torch.arange(k, device=device).repeat(num_tokens)[order]
            send_meta = torch.stack([
                ordered_global_eids.int(),
                torch.full_like(ordered_global_eids, self.rank, dtype=torch.int32),
                ordered_token_idx.int(),
                k_idx.int(),
                torch.zeros_like(ordered_global_eids, dtype=torch.int32)
            ], dim=1)

            # Pack metadata and data as float32 for efficient transfer
            send_meta_float = send_meta.float()
            send_data_float = send_data.float()
            combined_send = torch.cat([send_meta_float, send_data_float], dim=1)

        # Prepare receive buffer with wavefront alignment
        aligned_total_recv = self._align_to_wavefront(total_recv)
        recv_combined = torch.empty((aligned_total_recv, self.META_DIM + self.hidden_dim), dtype=torch.float32, device=device)

        # Single all_to_all_single for packed data
        if total_send > 0:
            dist.all_to_all_single(
                recv_combined[:total_recv],
                combined_send,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts_t.tolist()
            )

        # Handle empty receive case
        if total_recv == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            aligned_max_recv = self._align_to_wavefront(self.max_recv)
            expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, expert_x, expert_meta

        # Unpack received data efficiently
        recv_meta_float = recv_combined[:total_recv, :self.META_DIM]
        recv_data_float = recv_combined[:total_recv, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.in_dtype)

        # Group by local expert using sorting - vectorized approach
        global_eid_recv = recv_meta[:, 0].long()
        local_eid_flat = global_eid_recv.remainder(self.num_local_experts)

        # Sort by local expert id
        order_local = torch.argsort(local_eid_flat)
        sorted_recv_buf = recv_buf[order_local]
        sorted_recv_meta = recv_meta[order_local]
        sorted_local_eid = local_eid_flat[order_local]

        # Count tokens per expert
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).int()
        expert_num_tokens = local_counts.clone()

        # Allocate output tensors with wavefront alignment
        aligned_max_recv = self._align_to_wavefront(self.max_recv)
        expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim), 
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM), 
                                  dtype=torch.int32, device=device)

        # Vectorized distribution to experts
        if sorted_recv_buf.shape[0] > 0:
            split_sizes = local_counts.tolist()
            start_idx = 0
            for local_id, cnt in enumerate(split_sizes):
                if cnt > 0:
                    expert_x[local_id, :cnt].copy_(sorted_recv_buf[start_idx:start_idx+cnt])
                    expert_meta[local_id, :cnt].copy_(sorted_recv_meta[start_idx:start_idx+cnt])
                    start_idx += cnt

        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine ----------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, 
                expert_meta: torch.Tensor, expert_y: torch.Tensor, 
                expert_num_tokens: torch.Tensor):
        cfg = self.cfg
        device = out_tokens.device

        # Get counts and calculate total send size
        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)

        # Gather send entries (y + meta) using vectorized operations
        if total_send == 0:
            send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
            send_y = torch.empty((0, self.hidden_dim), dtype=cfg.out_dtype, device=device)
        else:
            # More efficient gathering using list comprehension
            valid_entries = [(i, cnt) for i, cnt in enumerate(counts) if cnt > 0]
            if valid_entries:
                meta_list = [expert_meta[i, :cnt] for i, cnt in valid_entries]
                y_list = [expert_y[i, :cnt] for i, cnt in valid_entries]
                send_meta = torch.cat(meta_list, dim=0)
                send_y = torch.cat(y_list, dim=0)
            else:
                send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
                send_y = torch.empty((0, self.hidden_dim), dtype=cfg.out_dtype, device=device)

        # Calculate destination ranks
        dst_ranks = send_meta[:, 1].long() if total_send > 0 else torch.empty((0,), dtype=torch.long, device=device)
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size) if total_send > 0 else torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        total_recv = recv_counts.sum().item()

        # Pack send data efficiently
        if total_send == 0:
            combined_send = torch.empty((0, self.META_DIM + self.hidden_dim), dtype=torch.float32, device=device)
        else:
            # Sort by destination rank
            order = torch.argsort(dst_ranks)
            ordered_meta = send_meta[order]
            ordered_y = send_y[order]
            
            # Pack as float32
            ordered_meta_float = ordered_meta.float()
            ordered_y_float = ordered_y.float()
            combined_send = torch.cat([ordered_meta_float, ordered_y_float], dim=1)

        # Prepare receive buffer with wavefront alignment
        aligned_total_recv = self._align_to_wavefront(total_recv)
        recv_combined = torch.empty((aligned_total_recv, self.META_DIM + self.hidden_dim), dtype=torch.float32, device=device)
        
        # Single all_to_all_single for packed data
        if total_send > 0:
            dist.all_to_all_single(
                recv_combined[:total_recv],
                combined_send,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts_t.tolist()
            )

        # Handle empty receive
        if total_recv == 0:
            return out_tokens

        # Unpack received data efficiently
        recv_meta_float = recv_combined[:total_recv, :self.META_DIM]
        recv_data_float = recv_combined[:total_recv, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.out_dtype)

        # Extract metadata and apply weights using vectorized operations
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()
        
        # Apply weights using vectorized operations
        w = weights[src_tokens, src_k].float().unsqueeze(1)
        recv_buf_float = recv_buf.float()
        weighted = recv_buf_float * w

        # Vectorized scatter_add for maximum performance
        idx = src_tokens.unsqueeze(1).expand(-1, self.hidden_dim)
        if out_tokens.dtype != torch.float32:
            out_acc = out_tokens.float()
            out_acc.scatter_add_(0, idx, weighted)
            out_tokens.copy_(out_acc.to(out_tokens.dtype))
        else:
            out_tokens.scatter_add_(0, idx, weighted)

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    
    # Create ultra high-performance AMD all-to-all instance
    ata = UltraHighPerformanceAMDOptimizedAllToAll(cfg, rank, world_size)

    # Dispatch tokens to experts
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Expert computation (simulated)
    # AMD optimization: Use float16 for better performance on AMD GPUs
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # Combine expert outputs back to tokens
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]