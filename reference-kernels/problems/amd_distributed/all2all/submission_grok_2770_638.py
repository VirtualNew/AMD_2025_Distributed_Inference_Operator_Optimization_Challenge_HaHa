import torch
import torch.distributed as dist
from task import input_t, output_t
import torch._inductor.config as inductor_config

# Set inductor configs for potential future use (disabled compile for now)
inductor_config.max_autotune = True
inductor_config.triton.autotune_pointwise = True
inductor_config.max_autotune_gemm = True
inductor_config.freezing = True
inductor_config.epilogue_fusion = True

# ---------------- Optimized Packed All2All Implementation ----------------
class OptimizedPackedAllToAll:
    META_DIM = 5  # [global_exp, src_rank, src_token, src_k, pad]

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden_dim = cfg.hidden_dim

    def _compute_destinations(self, indices: torch.Tensor):
        """
        Compute flattened destination information for tokens.
        
        Args:
            indices: (num_tokens, experts_per_token) global expert ids
            
        Returns:
            dst_flat: (num_tokens * experts_per_token,) destination ranks
            src_token_idx: (num_tokens * experts_per_token,) source token indices
            global_eids_flat: (num_tokens * experts_per_token,) global expert ids
        """
        device = indices.device
        num_tokens, k = indices.shape
        
        # Compute destination ranks for each token-expert pair
        dst_ranks = torch.div(indices, self.num_local_experts, rounding_mode='floor')
        dst_flat = dst_ranks.reshape(-1)
        
        # Create source token indices
        src_token_idx = torch.arange(num_tokens, device=device).repeat_interleave(k)
        global_eids_flat = indices.reshape(-1)
        
        return dst_flat, src_token_idx, global_eids_flat

    # ---------- dispatch: pack meta+data and one all_to_all_single ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Dispatch tokens to experts using optimized packed all-to-all.
        
        Returns:
            expert_num_tokens: (num_local_experts,) number of tokens per expert
            expert_x: (num_local_experts, max_recv, hidden_dim) token data
            expert_meta: (num_local_experts, max_recv, META_DIM) metadata
        """
        cfg = self.cfg
        device = dp_x.device
        num_tokens, k = indices.shape

        # Compute destination information
        dst_flat, src_token_idx, global_eids_flat = self._compute_destinations(indices)

        # Calculate send counts per rank
        send_counts = torch.bincount(dst_flat, minlength=self.world_size)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        
        total_send = send_counts.sum().item()
        total_recv = recv_counts.sum().item()

        per_row_len = self.META_DIM + self.hidden_dim

        # Pre-allocate send buffer
        combined_send = torch.empty((total_send, per_row_len), dtype=torch.float32, device=device) if total_send > 0 else torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        
        if total_send > 0:
            # Sort by destination for contiguous memory access
            order = torch.argsort(dst_flat)
            ordered_token_idx = src_token_idx[order]
            ordered_global_eids = global_eids_flat[order]
            
            # Gather data in order
            send_data = dp_x[ordered_token_idx]
            
            # Create metadata
            k_idx = torch.arange(k, device=device).repeat(num_tokens)[order]
            send_meta = torch.stack([
                ordered_global_eids.int(),
                torch.full_like(ordered_global_eids, self.rank, dtype=torch.int32),
                ordered_token_idx.int(),
                k_idx.int(),
                torch.zeros_like(ordered_global_eids, dtype=torch.int32)
            ], dim=1)

            # Pack metadata and data as float32 for efficient transfer
            combined_send[:, :self.META_DIM].copy_(send_meta.float())
            combined_send[:, self.META_DIM:].copy_(send_data.float())

        # Pre-allocate receive buffer
        recv_combined = torch.empty((total_recv, per_row_len), dtype=torch.float32, device=device)

        # Single all_to_all_single for packed data
        dist.all_to_all_single(
            recv_combined,
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
            return expert_num_tokens, expert_x, expert_meta

        # Unpack received data
        recv_meta_float = recv_combined[:, :self.META_DIM]
        recv_data_float = recv_combined[:, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.in_dtype)

        # Group by local expert using sorting
        global_eid_recv = recv_meta[:, 0].long()
        local_eid_flat = torch.remainder(global_eid_recv, self.num_local_experts).long()

        # Sort by local expert id
        order_local = torch.argsort(local_eid_flat)
        sorted_recv_buf = recv_buf[order_local]
        sorted_recv_meta = recv_meta[order_local]
        sorted_local_eid = local_eid_flat[order_local]

        # Count tokens per expert
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).int()
        expert_num_tokens = local_counts.clone()

        # Allocate output tensors
        expert_x = torch.empty((self.num_local_experts, self.max_recv, self.hidden_dim), 
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), 
                                  dtype=torch.int32, device=device)

        # Distribute data to experts using vectorized operations
        if sorted_recv_buf.shape[0] > 0:
            split_sizes = local_counts.tolist()
            start_idx = 0
            for local_id, cnt in enumerate(split_sizes):
                if cnt > 0:
                    end_idx = start_idx + cnt
                    expert_x[local_id, :cnt].copy_(sorted_recv_buf[start_idx:end_idx])
                    expert_meta[local_id, :cnt].copy_(sorted_recv_meta[start_idx:end_idx])
                    start_idx = end_idx

        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine: optimized packed all_to_all back ----------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, 
                expert_meta: torch.Tensor, expert_y: torch.Tensor, 
                expert_num_tokens: torch.Tensor):
        """
        Combine expert outputs back to tokens using optimized packed all-to-all.
        """
        cfg = self.cfg
        device = out_tokens.device

        # Get counts and calculate total send size
        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)

        # Pre-allocate send buffers
        send_meta = torch.empty((total_send, self.META_DIM), dtype=torch.int32, device=device) if total_send > 0 else torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
        send_y = torch.empty((total_send, self.hidden_dim), dtype=cfg.out_dtype, device=device) if total_send > 0 else torch.empty((0, self.hidden_dim), dtype=cfg.out_dtype, device=device)

        if total_send > 0:
            # Vectorized gathering of data
            meta_list = []
            y_list = []
            start_idx = 0
            for local_id, cnt in enumerate(counts):
                if cnt > 0:
                    end_idx = start_idx + cnt
                    meta_list.append(expert_meta[local_id, :cnt])
                    y_list.append(expert_y[local_id, :cnt])
                    start_idx = end_idx
            
            if meta_list:
                send_meta.copy_(torch.cat(meta_list, dim=0))
                send_y.copy_(torch.cat(y_list, dim=0))

        # Calculate destination ranks
        dst_ranks = send_meta[:, 1].long() if total_send > 0 else torch.empty((0,), dtype=torch.long, device=device)
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size) if total_send > 0 else torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts_t = send_counts.clone()
        
        # Exchange recv counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t
        total_recv = recv_counts.sum().item()

        per_row_len = self.META_DIM + self.hidden_dim

        # Pack send data
        combined_send = torch.empty((total_send, per_row_len), dtype=torch.float32, device=device) if total_send > 0 else torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        if total_send > 0:
            # Sort by destination rank
            order = torch.argsort(dst_ranks)
            ordered_meta = send_meta[order]
            ordered_y = send_y[order]
            
            # Pack as float32
            combined_send[:, :self.META_DIM].copy_(ordered_meta.float())
            combined_send[:, self.META_DIM:].copy_(ordered_y.float())

        # Prepare receive buffer
        recv_combined = torch.empty((total_recv, per_row_len), dtype=torch.float32, device=device)
        
        # Single all_to_all_single for packed data
        dist.all_to_all_single(
            recv_combined,
            combined_send,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts_t.tolist()
        )

        # Handle empty receive
        if total_recv == 0:
            return out_tokens

        # Unpack received data
        recv_meta_float = recv_combined[:, :self.META_DIM]
        recv_data_float = recv_combined[:, self.META_DIM:]
        recv_meta = recv_meta_float.int()
        recv_buf = recv_data_float.to(cfg.out_dtype)

        # Extract metadata
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()
        
        # Apply weights using vectorized operations
        w = weights[src_tokens, src_k].float().unsqueeze(1)
        weighted = recv_buf.float() * w

        # Vectorized scatter_add
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
    
    # Create optimized all-to-all instance
    ata = OptimizedPackedAllToAll(cfg, rank, world_size)

    # Dispatch tokens to experts
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Expert computation (simulated)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # Combine expert outputs back to tokens
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]