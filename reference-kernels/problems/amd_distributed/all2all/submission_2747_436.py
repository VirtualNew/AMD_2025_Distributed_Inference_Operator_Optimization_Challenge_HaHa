import torch
import torch.distributed as dist
from task import input_t, output_t

# ---------------- Packed All2All Implementation for AMD MI300x8 ----------------
class PackedAllToAll:
    META_DIM = 5  # [global_exp, src_rank, src_token, src_k, pad] stored as int32

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden = cfg.hidden_dim

    def _compute_flat_dest(self, indices: torch.Tensor):
        """
        indices: (num_tokens, k) global expert ids (int)
        returns:
          dst_flat: (num_tokens * k,) int64
          src_token_idx: (num_tokens * k,) int64
          global_eids_flat: (num_tokens*k,) int64
        """
        device = indices.device
        num_tokens, k = indices.shape
        dst_ranks = indices.div(self.num_local_experts, rounding_mode='floor').to(torch.int64)
        dst_flat = dst_ranks.reshape(-1)
        src_token_idx = torch.arange(num_tokens, device=device, dtype=torch.int64).repeat_interleave(k)
        global_eids_flat = indices.reshape(-1).to(torch.int64)
        return dst_flat, src_token_idx, global_eids_flat

    # ---------- dispatch: pack meta+data and one all_to_all_single ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        Returns:
          expert_num_tokens (num_local_experts, int32),
          expert_x (num_local_experts, max_recv, hidden) in cfg.in_dtype,
          expert_meta (num_local_experts, max_recv, META_DIM) int32
        Key idea: pack meta (as float32 header) + data row into single tensor, call one all_to_all.
        """
        cfg = self.cfg
        device = dp_x.device
        num_tokens, k = indices.shape

        # compute flattened destination info
        dst_flat, src_token_idx, global_eids_flat = self._compute_flat_dest(indices)

        # send_counts per rank
        send_counts = torch.bincount(dst_flat, minlength=self.world_size).to(torch.int64)
        send_counts_t = send_counts.clone()
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.to(torch.int64)
        total_send = int(send_counts.sum().item())
        total_recv = int(recv_counts.sum().item())

        per_row_len = self.META_DIM + cfg.hidden_dim

        # Prepare send buffer
        if total_send == 0:
            combined_send = torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        else:
            # group by dst via argsort for contiguous blocks
            order = torch.argsort(dst_flat)
            ordered_token_idx = src_token_idx[order]
            ordered_global_eids = global_eids_flat[order]
            ordered_dst = dst_flat[order]

            send_data = dp_x[ordered_token_idx]  # (total_send, hidden)
            # meta
            k_idx = (torch.arange(k, device=device, dtype=torch.int32).repeat(num_tokens))[order]
            send_meta = torch.stack([
                ordered_global_eids.to(torch.int32),
                torch.full((ordered_global_eids.shape[0],), self.rank, dtype=torch.int32, device=device),
                ordered_token_idx.to(torch.int32),
                k_idx.to(torch.int32),
                torch.zeros_like(ordered_global_eids, dtype=torch.int32)
            ], dim=1)  # (total_send, META_DIM)

            # pack meta + data into float32
            send_meta_float = send_meta.to(torch.float32)
            send_data_float = send_data.to(torch.float32)
            combined_send = torch.cat([send_meta_float, send_data_float], dim=1)  # shape = (total_send, per_row_len)

        # Prepare recv buffer always
        recv_combined = torch.empty((total_recv, per_row_len), dtype=torch.float32, device=device)

        # Now do all_to_all_single across all ranks
        dist.all_to_all_single(
            recv_combined,
            combined_send,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts_t.tolist()
        )

        # Unpack recv_combined
        if total_recv == 0:
            # no recv data
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, expert_x, expert_meta

        recv_meta_float = recv_combined[:, : self.META_DIM]
        recv_data_float = recv_combined[:, self.META_DIM :]
        recv_meta = recv_meta_float.to(torch.int32)
        recv_buf = recv_data_float.to(cfg.in_dtype)

        # group by local expert
        global_eid_recv = recv_meta[:, 0].to(torch.int64)
        local_eid_flat = (global_eid_recv % self.num_local_experts).to(torch.int64)

        order_local = torch.argsort(local_eid_flat)
        sorted_recv_buf = recv_buf[order_local]
        sorted_recv_meta = recv_meta[order_local]
        sorted_local_eid = local_eid_flat[order_local]

        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).to(torch.int32)
        expert_num_tokens = local_counts.clone()

        # allocate expert_x / expert_meta
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device)

        # 更优化的数据分发方式
        if sorted_recv_buf.shape[0] > 0:
            split_sizes = local_counts.tolist()
            start_idx = 0
            for local_id, cnt in enumerate(split_sizes):
                if cnt > 0:
                    expert_x[local_id, :cnt].copy_(sorted_recv_buf[start_idx:start_idx+cnt])
                    expert_meta[local_id, :cnt].copy_(sorted_recv_meta[start_idx:start_idx+cnt])
                    start_idx += cnt

        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine: pack meta+data and single all_to_all back ----------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, expert_meta: torch.Tensor, expert_y: torch.Tensor, expert_num_tokens: torch.Tensor):
        """
        Build send entries from local experts (select first n tokens per expert),
        pack meta+data as float32 combined rows, do one all_to_all, then vectorized scatter_add to out_tokens.
        """
        cfg = self.cfg
        device = out_tokens.device

        counts = expert_num_tokens.to(torch.int64).tolist()
        total_send = sum(counts)

        # gather send entries (y + meta)
        if total_send == 0:
            send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
            send_y = torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
        else:
            # 优化的数据收集方式
            meta_list = []
            y_list = []
            for local_id, cnt in enumerate(counts):
                if cnt > 0:
                    meta_list.append(expert_meta[local_id, :cnt])
                    y_list.append(expert_y[local_id, :cnt])
            
            if meta_list:
                send_meta = torch.cat(meta_list, dim=0)  # (total_send, META_DIM)
                send_y = torch.cat(y_list, dim=0)        # (total_send, hidden)
            else:
                send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
                send_y = torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)

        dst_ranks = send_meta[:, 1].to(torch.int64) if total_send > 0 else torch.empty((0,), dtype=torch.int64, device=device)
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size).to(torch.int64) if total_send > 0 else torch.zeros((self.world_size,), dtype=torch.int64, device=device)
        send_counts_t = send_counts.clone()
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.to(torch.int64)
        total_recv = int(recv_counts.sum().item())

        per_row_len = self.META_DIM + cfg.hidden_dim

        # pack combined_send
        if total_send == 0:
            combined_send = torch.empty((0, per_row_len), dtype=torch.float32, device=device)
        else:
            order = torch.argsort(dst_ranks)
            ordered_meta = send_meta[order]
            ordered_y = send_y[order]
            ordered_meta_float = ordered_meta.to(torch.float32)
            ordered_y_float = ordered_y.to(torch.float32)
            combined_send = torch.cat([ordered_meta_float, ordered_y_float], dim=1)  # (total_send, per_row_len)

        recv_combined = torch.empty((total_recv, per_row_len), dtype=torch.float32, device=device)
        dist.all_to_all_single(
            recv_combined,
            combined_send,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts_t.tolist()
        )

        if total_recv == 0:
            return out_tokens

        recv_meta_float = recv_combined[:, : self.META_DIM]
        recv_data_float = recv_combined[:, self.META_DIM :]
        recv_meta = recv_meta_float.to(torch.int32)
        recv_buf = recv_data_float.to(cfg.out_dtype)

        src_tokens = recv_meta[:, 2].to(torch.int64)
        src_k = recv_meta[:, 3].to(torch.int64)
        w = weights[src_tokens, src_k].to(torch.float32).unsqueeze(1)  # (total_recv,1)
        recv_buf_float = recv_buf.to(torch.float32)
        weighted = recv_buf_float * w  # (total_recv, hidden)

        # 优化的scatter_add操作
        if out_tokens.dtype != torch.float32:
            out_acc = out_tokens.to(torch.float32)
            idx = src_tokens.unsqueeze(1).expand(-1, cfg.hidden_dim)
            out_acc.scatter_add_(0, idx, weighted)
            out_tokens.copy_(out_acc.to(out_tokens.dtype))
        else:
            idx = src_tokens.unsqueeze(1).expand(-1, cfg.hidden_dim)
            out_tokens.scatter_add_(0, idx, weighted)

        return out_tokens

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    ata = PackedAllToAll(cfg, rank, world_size)

    # dispatch
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # expert compute (模拟)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # combine
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=rank_data.x.device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[: rank_data.num_tokens]