import torch
import torch.distributed as dist
from task import input_t, output_t


class HighPerformanceAllToAll:
    """
    High performance implementation focused on simplicity and performance.
    """
    META_DIM = 5  # [global_eid, src_rank, src_token_idx, src_k, pad]

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden_dim = cfg.hidden_dim
        self.alignment = 128

    @staticmethod
    def _align(n, a):
        """Align dimension to specified alignment factor."""
        return ((n + a - 1) // a) * a

    def _compute_flat_info(self, indices: torch.Tensor):
        """
        Compute flattened destination information for tokens.
        """
        device = indices.device
        num_tokens, k = indices.shape
        
        # Compute destination ranks for each token-expert pair
        dst = (indices // self.num_local_experts).reshape(-1).long()
        
        # Create source token indices
        src_token = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, k).reshape(-1).long()
        global_eid = indices.reshape(-1).long()
        src_k = torch.arange(k, device=device).unsqueeze(0).expand(num_tokens, -1).reshape(-1).long()
        
        return dst, src_token, global_eid, src_k

    def _bincount(self, x, minlength):
        """Simple bincount implementation."""
        if x.numel() == 0:
            return torch.zeros(minlength, dtype=torch.long, device=x.device)
        counts = torch.zeros(minlength, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.long))
        return counts

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Dispatch tokens to experts using optimized separated transmission.
        """
        cfg = self.cfg
        device = dp_x.device

        # Ensure contiguous memory layout
        dp_x = dp_x.contiguous()
        indices = indices.contiguous()

        num_tokens, k = indices.shape

        # Compute flat information for all token-expert pairs
        dst_flat, src_token_idx, global_eids_flat, src_k_flat = self._compute_flat_info(indices)

        # Calculate send counts per rank
        send_counts = self._bincount(dst_flat, self.world_size).long()
        total_send = int(send_counts.sum().item())

        # Fast path: nothing to send
        if total_send == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            aligned_max_recv = self._align(self.max_recv, self.alignment)
            expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, expert_x, expert_meta

        # Sort by destination to create contiguous send buffer per rank
        sorted_indices = torch.argsort(dst_flat)
        ordered_dst = dst_flat[sorted_indices]
        ordered_src_token = src_token_idx[sorted_indices]
        ordered_global_eid = global_eids_flat[sorted_indices]
        ordered_src_k = src_k_flat[sorted_indices]
        ordered_data = dp_x[ordered_src_token]

        # Construct send metadata and send data
        send_meta = torch.empty((ordered_global_eid.shape[0], 5), dtype=torch.int32, device=device)
        send_meta[:, 0] = ordered_global_eid.to(torch.int32)
        send_meta[:, 1] = self.rank
        send_meta[:, 2] = ordered_src_token.to(torch.int32)
        send_meta[:, 3] = ordered_src_k.to(torch.int32)
        send_meta[:, 4] = 0
        
        # Ensure contiguous memory layout
        send_meta = send_meta.contiguous()
        ordered_data = ordered_data.contiguous()
        
        # Convert data type if needed
        if ordered_data.dtype != cfg.in_dtype:
            send_data = ordered_data.to(cfg.in_dtype)
        else:
            send_data = ordered_data

        # Exchange receive counts
        recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts, send_counts)
        total_recv = int(recv_counts.sum().item())

        # Align receive buffer size
        aligned_recv = self._align(total_recv, self.alignment) if total_recv > 0 else self.alignment

        # Pre-allocate receive buffers
        recv_meta = torch.empty((aligned_recv, self.META_DIM), dtype=torch.int32, device=device)
        recv_data = torch.empty((aligned_recv, self.hidden_dim), dtype=cfg.in_dtype, device=device)

        # Separate all_to_all_single calls for meta and data
        if total_send > 0 and total_recv > 0:
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
            return expert_num_tokens, expert_x, expert_meta

        # Trim buffers to actual received size
        if total_recv < aligned_recv:
            recv_meta = recv_meta[:total_recv]
            recv_data = recv_data[:total_recv]

        # Compute local expert id for each received record
        global_eid_recv = recv_meta[:, 0].long()
        local_eid = (global_eid_recv % self.num_local_experts).long()

        # Sort by local expert id to group items by local expert
        sorted_local_indices = torch.argsort(local_eid)
        sorted_local_eid = local_eid[sorted_local_indices]
        sorted_meta = recv_meta[sorted_local_indices]
        sorted_data = recv_data[sorted_local_indices]

        # Compute counts per local expert
        local_counts = self._bincount(sorted_local_eid, self.num_local_experts).long()
        expert_num_tokens = local_counts.to(torch.int32)

        # Allocate aligned expert buffers
        aligned_max_recv = self._align(self.max_recv, self.alignment)
        expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                  dtype=torch.int32, device=device)

        # If there are received items, compute per-record position within each expert group
        if total_recv > 0:
            # Cumulative counts => end indices in sorted array for each expert
            cum = torch.cumsum(local_counts, dim=0)
            # Start indices in sorted array for each expert
            starts = torch.empty_like(cum)
            starts[0] = 0
            starts[1:] = cum[:-1]
            # Index in sorted array for each item
            idx_in_sorted = torch.arange(total_recv, device=device)
            # Position within expert group for each item
            group_pos = idx_in_sorted - starts[sorted_local_eid]
            # Place sorted data into expert buffers
            expert_x[sorted_local_eid, group_pos] = sorted_data
            expert_meta[sorted_local_eid, group_pos] = sorted_meta

        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor,
                expert_meta: torch.Tensor, expert_y: torch.Tensor,
                expert_num_tokens: torch.Tensor):
        """
        Combine expert outputs back to tokens.
        """
        cfg = self.cfg
        device = out_tokens.device

        # Ensure contiguous memory layout
        out_tokens = out_tokens.contiguous()
        weights = weights.contiguous()
        expert_meta = expert_meta.contiguous()
        expert_y = expert_y.contiguous()
        expert_num_tokens = expert_num_tokens.contiguous()

        # Get counts and calculate total send size
        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)
        
        # Fast path: nothing to send
        if total_send == 0:
            return out_tokens

        # Gather valid entries from experts
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
        send_counts = self._bincount(dst_ranks, self.world_size).long()

        # Sort by destination to make contiguous per-dst block
        sorted_dst_indices = torch.argsort(dst_ranks)
        ordered_meta = send_meta[sorted_dst_indices]
        ordered_y = send_y[sorted_dst_indices]

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

        # Separate all_to_all_single calls for meta and data
        if total_send > 0 and total_recv > 0:
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
        if total_recv < aligned_recv:
            recv_meta = recv_meta[:total_recv]
            recv_y = recv_y[:total_recv]

        # Apply weights and scatter_add to out_tokens
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()

        # Get weights and apply them
        w = weights[src_tokens, src_k].to(recv_y.dtype).unsqueeze(1)
        weighted = recv_y.mul(w)  # (total_recv, hidden_dim)

        # Vectorized scatter_add operation
        expanded_indices = src_tokens.unsqueeze(1).expand(-1, weighted.shape[1])
        out_tokens.scatter_add_(0, expanded_indices, weighted)

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    # Unpack data
    cfg, rank_data, rank, world_size = data
    
    # Get device
    device = rank_data.x.device

    # Create high performance all-to-all instance
    ata = HighPerformanceAllToAll(cfg, int(rank), int(world_size))

    # Dispatch tokens to experts
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Simulated expert compute (use out_dtype)
    if expert_x.dtype != cfg.out_dtype:
        expert_y = expert_x.to(cfg.out_dtype)
    else:
        expert_y = expert_x
    expert_y.mul_(1 + int(rank))

    # Combine expert outputs back to tokens
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:int(rank_data.num_tokens)]