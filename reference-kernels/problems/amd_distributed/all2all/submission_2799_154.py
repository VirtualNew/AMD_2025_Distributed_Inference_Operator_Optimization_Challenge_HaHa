import torch
import torch.distributed as dist
from task import input_t, output_t

# ---------------- Optimized All2All pytorch impl ----------------
class OptimizedAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # number of local experts per rank
        self.num_local_experts = cfg.num_experts // world_size
        # maximum receive capacity per expert across all DP (worst-case)
        self.max_recv = cfg.max_num_tokens * world_size
        self.device = torch.device("cuda", rank)

    # ---------- dispatch (vectorized) ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        """
        dp_x: (num_tokens, hidden_dim) - dtype = cfg.in_dtype
        indices: (num_tokens, experts_per_token) - long (global expert id)
        returns:
            expert_num_tokens: (num_local_experts,) int32
            expert_x: (num_local_experts, max_recv, hidden_dim)
            expert_meta: (num_local_experts, max_recv, META_DIM) int32
        """
        cfg = self.cfg
        device = dp_x.device

        num_tokens = indices.shape[0]
        k = indices.shape[1]  # experts_per_token

        # ---- 1. compute destination ranks for each token-expert pair (vectorized) ----
        # dst_ranks: (num_tokens, k)
        dst_ranks = indices.div(self.num_local_experts, rounding_mode='floor').to(torch.int64).to(device)
        # flatten
        dst_flat = dst_ranks.reshape(-1)  # (num_tokens * k,)
        # repeat token indices to align with flattened expert entries
        src_token_idx = torch.arange(num_tokens, device=device, dtype=torch.int64).repeat_interleave(k)

        # compute send_counts via bincount
        send_counts = torch.bincount(dst_flat, minlength=self.world_size).to(torch.int64)
        # exchange send_counts to get recv_counts
        send_counts_t = send_counts.clone()
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        # all ranks exchange counts
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.to(torch.int64)

        # ---- 2. build send order: group flattened entries by dst rank ----
        # argsort by dst_flat -> stable grouping by dst rank
        if dst_flat.numel() == 0:
            # corner case: no sends
            send_buf = torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
            send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
        else:
            order = torch.argsort(dst_flat)
            ordered_token_idx = src_token_idx[order]  # token index repeatedly per k
            # create send_buf by indexing dp_x
            send_buf = dp_x[ordered_token_idx]  # (total_send, hidden_dim)
            # compose metadata for each flattened entry: [global_eid, src_rank, src_token, src_k, 0]
            # Need global_eid and src_k aligned to flattened order:
            global_eids_flat = indices.reshape(-1)[order].to(torch.int32)
            # compute src_k: (0..k-1) repeated
            src_k_flat = (torch.arange(k, device=device, dtype=torch.int32).repeat(num_tokens))[order]
            src_rank_arr = torch.full((global_eids_flat.shape[0],), self.rank, dtype=torch.int32, device=device)
            src_token_arr = ordered_token_idx.to(torch.int32)
            pad_arr = torch.zeros_like(global_eids_flat, dtype=torch.int32)
            send_meta = torch.stack([global_eids_flat, src_rank_arr, src_token_arr, src_k_flat, pad_arr], dim=1)  # (total_send, META_DIM)

        total_recv = int(recv_counts.sum().item())
        # prepare recv buffers
        recv_buf = torch.empty((total_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty((total_recv, self.META_DIM), dtype=torch.int32, device=device)

        # if we have sends (may be zero for some ranks)
        send_buf_for_all = send_buf
        send_meta_for_all = send_meta

        # Input split sizes: send_counts.tolist(), output split sizes: recv_counts.tolist()
        # all_to_all_single expects contiguous 1D tensors for meta, so we flatten meta
        # but for data we send as rows (2D) and rely on split sizes
        if send_buf_for_all.numel() == 0:
            # build empty input to avoid all_to_all complaining
            # create empty tensor of shape (0, hidden_dim) with correct dtype
            empty_send_buf = torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
            dist.all_to_all_single(recv_buf, empty_send_buf, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts_t.tolist())
            # send meta
            empty_send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
            dist.all_to_all_single(recv_meta.view(-1), empty_send_meta.view(-1),
                                   output_split_sizes=[c * self.META_DIM for c in recv_counts.tolist()],
                                   input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()])
        else:
            dist.all_to_all_single(recv_buf, send_buf_for_all, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts_t.tolist())
            dist.all_to_all_single(recv_meta.view(-1), send_meta_for_all.view(-1),
                                   output_split_sizes=[c * self.META_DIM for c in recv_counts.tolist()],
                                   input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()])
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # ---- 3. write tokens to each local expert (vectorized by sorting by local_eid) ----
        # local_eid = global_eid % num_local_experts
        if total_recv == 0:
            # empty
            expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
            expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim),
                                   dtype=cfg.in_dtype, device=device)
            expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM),
                                      dtype=torch.int32, device=device)
            return expert_num_tokens, expert_x, expert_meta

        global_eid_recv = recv_meta[:, 0].to(torch.int64)
        local_eid_flat = (global_eid_recv % self.num_local_experts).to(torch.int64)

        # sort by local_eid to pack tokens per local expert
        order_local = torch.argsort(local_eid_flat)
        sorted_recv_buf = recv_buf[order_local]
        sorted_recv_meta = recv_meta[order_local]
        sorted_local_eid = local_eid_flat[order_local]

        # counts per local expert
        local_counts = torch.bincount(sorted_local_eid, minlength=self.num_local_experts).to(torch.int32)
        expert_num_tokens = local_counts.clone()

        # allocate expert_x and expert_meta once
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device)

        # fill expert_x / expert_meta by split according to local_counts
        if sorted_recv_buf.shape[0] > 0:
            # compute split sizes
            split_sizes = local_counts.tolist()
            # split the sorted buffers into list of tensors per expert (length world_size_local_experts)
            # For efficiency, use torch.split; this creates views (cheap)
            splitted_bufs = torch.split(sorted_recv_buf, split_sizes, dim=0)
            splitted_metas = torch.split(sorted_recv_meta, split_sizes, dim=0)
            # Now write into expert_x and expert_meta
            for local_id, cnt in enumerate(split_sizes):
                if cnt == 0:
                    # initialize with zeros or leave uninitialized
                    continue
                expert_x[local_id, :cnt].copy_(splitted_bufs[local_id])
                expert_meta[local_id, :cnt].copy_(splitted_metas[local_id])
        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine (vectorized) ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # (max_num_tokens, hidden_dim) output, dtype = cfg.out_dtype
        weights: torch.Tensor,  # (num_tokens, experts_per_token)
        expert_meta: torch.Tensor,  # (num_local_experts, max_recv, META_DIM)
        expert_y: torch.Tensor,  # (num_local_experts, max_recv, hidden_dim) dtype=cfg.out_dtype
        expert_num_tokens: torch.Tensor,  # (num_local_experts,)
    ):
        """
        Vectorized collection from local experts -> all_to_all -> scatter_add to out_tokens.
        """
        device = out_tokens.device
        cfg = self.cfg
        num_local = self.num_local_experts

        # 1. extract valid entries per local expert and flatten them (vectorized)
        # For each local_eid, take first expert_num_tokens[local_eid] rows from expert_y and expert_meta
        counts = expert_num_tokens.to(torch.int64).tolist()
        if sum(counts) == 0:
            # nothing to send
            return out_tokens

        # gather the useful rows into flat tensors
        chunks_y = []
        chunks_meta = []
        for local_id, cnt in enumerate(counts):
            if cnt == 0:
                continue
            chunks_y.append(expert_y[local_id, :cnt])  # (cnt, hidden_dim)
            chunks_meta.append(expert_meta[local_id, :cnt])  # (cnt, META_DIM)

        send_y = torch.cat(chunks_y, dim=0)  # (total_send, hidden_dim)
        send_meta = torch.cat(chunks_meta, dim=0).to(torch.int32)  # (total_send, META_DIM)

        # 2. compute send_counts by destination ranks (dst_rank is in meta[:,1])
        dst_ranks = send_meta[:, 1].to(torch.int64)
        send_counts = torch.bincount(dst_ranks, minlength=self.world_size).to(torch.int64)
        send_counts_t = send_counts.clone()
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        recv_counts = recv_counts_t.to(torch.int64)

        # 3. order send entries by dst_rank (argsort) to prepare for all_to_all
        order = torch.argsort(dst_ranks)
        if send_y.numel() == 0:
            send_buf = torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
            send_meta_flat = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
        else:
            send_buf = send_y[order]
            send_meta_flat = send_meta[order]

        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty((total_recv, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty((total_recv, self.META_DIM), dtype=torch.int32, device=device)

        # do all_to_all
        if send_buf.numel() == 0:
            empty_send_buf = torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
            dist.all_to_all_single(recv_buf, empty_send_buf,
                                   output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts_t.tolist())
            empty_send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)
            dist.all_to_all_single(recv_meta.view(-1), empty_send_meta.view(-1),
                                   output_split_sizes=[c * self.META_DIM for c in recv_counts.tolist()],
                                   input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()])
        else:
            dist.all_to_all_single(recv_buf, send_buf, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts_t.tolist())
            dist.all_to_all_single(recv_meta.view(-1), send_meta_flat.view(-1),
                                   output_split_sizes=[c * self.META_DIM for c in recv_counts.tolist()],
                                   input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()])

        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 4. vectorized write-back: for each recv entry, we have src_token (col2) and src_k (col3)
        # compute weights lookup, multiply recv_buf by weight, then scatter_add into out_tokens at src_token index
        if recv_buf.numel() == 0:
            return out_tokens

        src_tokens = recv_meta[:, 2].to(torch.int64)  # (total_recv,)
        src_k = recv_meta[:, 3].to(torch.int64)  # (total_recv,)

        # gather weights for each recv entry from weights[src_token, src_k]
        # weights dtype may differ; convert to float for accumulation
        w = weights[src_tokens, src_k].to(torch.float32).unsqueeze(1)  # (total_recv, 1)
        # ensure recv_buf in float32 for accumulation if needed
        recv_buf_float = recv_buf.to(torch.float32)

        weighted = recv_buf_float * w  # (total_recv, hidden_dim)

        # scatter_add into out_tokens (out_tokens dtype might be cfg.out_dtype, but accumulate in float)
        # expand src_tokens to match hidden_dim for scatter_add
        idx = src_tokens.unsqueeze(1).expand(-1, cfg.hidden_dim)  # (total_recv, hidden_dim)
        # out_tokens may be dtype out_dtype; do accumulation in float32 then cast back if needed
        if out_tokens.dtype != torch.float32:
            # accumulate into a float32 buffer then copy back
            out_acc = out_tokens.to(torch.float32)
            out_acc.scatter_add_(0, idx, weighted)
            out_tokens.copy_(out_acc.to(out_tokens.dtype))
        else:
            out_tokens.scatter_add_(0, idx, weighted)

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    """
    data is a tuple: (cfg, rank_data, rank, world_size)
    rank_data contains: num_tokens, indices, weights, x
    """
    cfg, rank_data, rank, world_size = data
    # set current cuda device
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    ata = OptimizedAllToAll(cfg, rank, world_size)

    # dispatch
    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)

    # simple expert computation simulation: convert expert_x -> expert_y
    # In real kernel, this would be a per-expert fwd (e.g., matmul + activation)
    # Here we use the baseline toy op: expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # prepare out tensor (max_num_tokens x hidden_dim)
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=rank_data.x.device)

    # combine
    ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    # return only valid tokens
    return y[: rank_data.num_tokens]