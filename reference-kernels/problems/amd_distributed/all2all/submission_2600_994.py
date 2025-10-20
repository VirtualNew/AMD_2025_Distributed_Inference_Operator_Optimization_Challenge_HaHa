import torch
import torch.distributed as dist
from task import input_t, output_t

# ---------------- 高性能 PackedAllToAllV3 ----------------
class PackedAllToAllV3:
    """
    进一步优化版本：
    - 分离 meta (int32) / data (floatX) 的传输
    - 排序后构造连续发送区，利用 input_split_sizes/output_split_sizes
    - 收到后对 local_expert 重新排序并用 vectorized advanced indexing 写入 expert buffers
    - 最小化临时内存与 dtype 转换
    """
    META_DIM = 5  # [global_eid, src_rank, src_token_idx, src_k, pad]

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        self.hidden_dim = cfg.hidden_dim

        # 对齐参数：根据硬件调优
        self.alignment = 128

    @staticmethod
    def _align(n, a):
        return ((n + a - 1) // a) * a

    @staticmethod
    def _contig(t: torch.Tensor):
        return t.contiguous() if not t.is_contiguous() else t

    def _compute_flat_info(self, indices: torch.Tensor):
        # indices: (num_tokens, k)
        num_tokens, k = indices.shape
        device = indices.device
        dst = torch.div(indices, self.num_local_experts, rounding_mode='floor').reshape(-1).long()  # (num_tokens*k,)
        src_token = torch.arange(num_tokens, device=device).repeat_interleave(k).long()
        global_eid = indices.reshape(-1).long()
        src_k = torch.arange(k, device=device).repeat(num_tokens).long()
        return dst, src_token, global_eid, src_k

    # ---------------- Dispatch ----------------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        dp_x: (num_tokens, hidden_dim) in_dtype
        indices: (num_tokens, k) global expert ids (long)
        returns:
            expert_num_tokens: (num_local_experts,) int32
            expert_x: (num_local_experts, aligned_max_recv, hidden_dim) in_dtype
            expert_meta: (num_local_experts, aligned_max_recv, META_DIM) int32
        """
        cfg = self.cfg
        device = dp_x.device

        dp_x = self._contig(dp_x)
        indices = self._contig(indices)

        num_tokens, k = indices.shape

        # compute flat infos
        dst_flat, src_token_idx, global_eids_flat, src_k_flat = self._compute_flat_info(indices)

        # send counts per rank
        send_counts = torch.bincount(dst_flat, minlength=self.world_size).long().to(device)
        total_send = int(send_counts.sum().item())

        # fast path: nothing to send
        if total_send == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            aligned_max_recv = self._align(self.max_recv, self.alignment)
            expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

        # order by dst to create contiguous send buffer per rank
        order = torch.argsort(dst_flat)
        ordered_dst = dst_flat[order]  # (total_send,)
        ordered_src_token = src_token_idx[order]
        ordered_global_eid = global_eids_flat[order]
        ordered_src_k = src_k_flat[order]
        ordered_data = dp_x[ordered_src_token]  # (total_send, hidden_dim)

        # construct send meta (int32) and send data (in_dtype)
        send_meta = torch.stack([
            ordered_global_eid.to(torch.int32),
            torch.full_like(ordered_global_eid, fill_value=self.rank, dtype=torch.int32),
            ordered_src_token.to(torch.int32),
            ordered_src_k.to(torch.int32),
            torch.zeros_like(ordered_global_eid, dtype=torch.int32)
        ], dim=1)  # (total_send, META_DIM)
        send_meta = self._contig(send_meta)
        send_data = self._contig(ordered_data.to(cfg.in_dtype))

        # exchange recv counts
        recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts, send_counts)
        total_recv = int(recv_counts.sum().item())

        aligned_recv = self._align(total_recv, self.alignment) if total_recv > 0 else 0

        # allocate recv buffers
        recv_meta = torch.empty((aligned_recv, self.META_DIM), dtype=torch.int32, device=device)
        recv_data = torch.empty((aligned_recv, self.hidden_dim), dtype=cfg.in_dtype, device=device)

        # all_to_all_single for meta and data separately (keeps dtype correct, avoids casts)
        if total_send > 0:
            dist.all_to_all_single(recv_meta[:total_recv], send_meta,
                                   output_split_sizes=recv_counts.tolist(),
                                   input_split_sizes=send_counts.tolist())
            dist.all_to_all_single(recv_data[:total_recv], send_data,
                                   output_split_sizes=recv_counts.tolist(),
                                   input_split_sizes=send_counts.tolist())

        if total_recv == 0:
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            aligned_max_recv = self._align(self.max_recv, self.alignment)
            expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

        recv_meta = recv_meta[:total_recv]
        recv_data = recv_data[:total_recv]

        # compute local expert id for each received record
        global_eid_recv = recv_meta[:, 0].long()
        local_eid = (global_eid_recv % self.num_local_experts).long()  # (total_recv,)

        # sort by local_eid to group items by local expert
        order_local = torch.argsort(local_eid)
        sorted_local_eid = local_eid[order_local]
        sorted_meta = recv_meta[order_local]
        sorted_data = recv_data[order_local]

        # compute counts per local expert
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).long().to(device)
        expert_num_tokens = local_counts.to(torch.int32)

        # allocate aligned expert buffers
        aligned_max_recv = self._align(self.max_recv, self.alignment)
        expert_x = torch.empty((self.num_local_experts, aligned_max_recv, self.hidden_dim),
                               dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, aligned_max_recv, self.META_DIM),
                                  dtype=torch.int32, device=device)

        # If there's any received item, compute per-record position within each expert group
        if total_recv > 0:
            # cumulative counts => end indices in sorted array for each expert
            cum = torch.cumsum(local_counts, dim=0)  # (num_local_experts,)
            # start indices in sorted array for each expert
            starts = torch.empty_like(cum)
            starts[0] = 0
            starts[1:] = cum[:-1]
            # idx_in_sorted = 0..total_recv-1
            idx_in_sorted = torch.arange(total_recv, device=device)
            # group_pos = idx_in_sorted - starts[sorted_local_eid]
            group_pos = idx_in_sorted - starts[sorted_local_eid]  # (total_recv,)
            # Now place sorted_meta and sorted_data into expert buffers at (expert_id, group_pos)
            # advanced indexing write (vectorized)
            expert_x[sorted_local_eid, group_pos] = sorted_data
            expert_meta[sorted_local_eid, group_pos] = sorted_meta

        return expert_num_tokens, self._contig(expert_x), self._contig(expert_meta)

    # ---------------- Combine ----------------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor,
                expert_meta: torch.Tensor, expert_y: torch.Tensor,
                expert_num_tokens: torch.Tensor):
        """
        Combine expert outputs back to tokens.
        out_tokens: (max_num_tokens, hidden_dim) out_dtype
        weights: (max_num_tokens, k) float
        expert_meta: (num_local_experts, aligned_max_recv, META_DIM) int32
        expert_y: (num_local_experts, aligned_max_recv, hidden_dim) out_dtype
        expert_num_tokens: (num_local_experts,) int32
        """
        cfg = self.cfg
        device = out_tokens.device

        out_tokens = self._contig(out_tokens)
        weights = self._contig(weights)
        expert_meta = self._contig(expert_meta)
        expert_y = self._contig(expert_y)
        expert_num_tokens = self._contig(expert_num_tokens)

        counts = expert_num_tokens.long().tolist()
        total_send = sum(counts)
        if total_send == 0:
            return out_tokens

        # gather valid entries from experts (concatenate)
        meta_list = []
        y_list = []
        for local_id, cnt in enumerate(counts):
            if cnt > 0:
                meta_list.append(expert_meta[local_id, :cnt])
                y_list.append(expert_y[local_id, :cnt])
        send_meta = torch.cat(meta_list, dim=0)  # (total_send, META_DIM) int32
        send_y = torch.cat(y_list, dim=0)        # (total_send, hidden_dim) out_dtype

        dst_ranks = send_meta[:, 1].long()
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size).long().to(device)

        # sort by dst to make contiguous per-dst block
        order = torch.argsort(dst_ranks)
        ordered_meta = send_meta[order]
        ordered_y = send_y[order]

        # exchange recv counts
        recv_counts = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts, send_counts)
        total_recv = int(recv_counts.sum().item())
        if total_recv == 0:
            return out_tokens

        aligned_recv = self._align(total_recv, self.alignment)
        recv_meta = torch.empty((aligned_recv, self.META_DIM), dtype=torch.int32, device=device)
        recv_y = torch.empty((aligned_recv, self.hidden_dim), dtype=cfg.out_dtype, device=device)

        dist.all_to_all_single(recv_meta[:total_recv], ordered_meta,
                               output_split_sizes=recv_counts.tolist(),
                               input_split_sizes=send_counts.tolist())
        dist.all_to_all_single(recv_y[:total_recv], ordered_y,
                               output_split_sizes=recv_counts.tolist(),
                               input_split_sizes=send_counts.tolist())

        recv_meta = recv_meta[:total_recv]
        recv_y = recv_y[:total_recv]

        # apply weights and scatter_add to out_tokens
        src_tokens = recv_meta[:, 2].long()
        src_k = recv_meta[:, 3].long()

        # w shape (total_recv, 1) in recv_y dtype
        w = weights[src_tokens, src_k].to(recv_y.dtype).unsqueeze(1)
        weighted = recv_y * w  # (total_recv, hidden_dim)

        idx = src_tokens.unsqueeze(1).expand(-1, self.hidden_dim)  # (total_recv, hidden_dim)
        if out_tokens.dtype != weighted.dtype:
            out_acc = out_tokens.to(weighted.dtype)
            out_acc.scatter_add_(0, idx, weighted)
            out_tokens.copy_(out_acc.to(out_tokens.dtype))
        else:
            out_tokens.scatter_add_(0, idx, weighted)

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    device = rank_data.x.device

    ata = PackedAllToAllV3(cfg, rank, world_size)

    # dispatch
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # simulated expert compute (use out_dtype)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # combine
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    y = ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[:rank_data.num_tokens]
