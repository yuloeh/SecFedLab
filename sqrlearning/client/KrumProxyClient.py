import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

from .BackdoorClient import BackdoorClient


class KrumProxyClient(BackdoorClient):
    """
    Krum-Proxy (Subramanian et al., 2026) 按 Algorithm 1 实现。

    每轮流程：
      1. 本地干净训练 → Δ_loc（用于 L_norm 和投影下限）
      2. 用不相交分区训练 R 个随机参考 → {Δ_r}（每轮重建）
      3. 锚点 Δ_anchor = 0.85Δ_(1) + 0.15Δ_(2)
      4. Stage 1：后门注入（仅 CE）
      5. Stage 2：几何塑造（CE + norm + krum proxy + anchor）
      6. 投影：σ_ref 插值 + 范数匹配
    """

    def __init__(self, *args,
                 num_references: int = 6,
                 krum_k: int,
                 ref_lr_range: tuple = (0.005, 0.02),
                 ref_epochs_range: tuple = (1, 3),
                 stage1_epochs: int = 1,
                 stage2_epochs: int = 1,
                 lam_l: float = 0.4,
                 lam_n: float = 0.1,
                 lam_k: float = 0.25,
                 lam_a: float = 0.05,
                 clip_tau: float = 5.0,
                 seed: int | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_references = num_references
        self.krum_k = krum_k
        self.ref_lr_range = ref_lr_range
        self.ref_epochs_range = ref_epochs_range
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.lam_l, self.lam_n, self.lam_k, self.lam_a = lam_l, lam_n, lam_k, lam_a
        self.clip_tau = clip_tau
        self.seed = seed

        # 固定参数名列表，保证所有扁平化顺序一致
        self.param_names = [n for n, p in self.net.named_parameters() if p.is_floating_point()]

        # 不相交分区只划一次；参考训练每轮从 w_t 出发
        rng = np.random.RandomState(seed) if seed is not None else np.random
        indices = np.arange(len(self.train_data))
        rng.shuffle(indices)
        self.partitions = [Subset(self.train_data, p.tolist())
                           for p in np.array_split(indices, num_references)]

    # ---------- 向量化工具 ----------
    def _flatten(self, delta, detach=True):
        parts = []
        for n in self.param_names:
            if n in delta:
                t = delta[n]
                parts.append((t.detach() if detach else t).view(-1))
        return torch.cat(parts)

    def _current_delta_flat(self, w_t):
        """从参数直接取，保留计算图"""
        parts = []
        for n, p in self.net.named_parameters():
            if n in w_t and p.is_floating_point():
                parts.append((p - w_t[n].detach()).view(-1))
        return torch.cat(parts)

    def _state_delta(self, w_new, w_t):
        return {k: (w_new[k] - w_t[k]).clone() for k in w_t}

    def _sq_dist_flat(self, a, b):
        return ((a - b) ** 2).sum()

    # ---------- 每轮重建参考集 ----------
    def _build_references(self, w_t):
        refs = []
        for partition in self.partitions:
            lr = np.random.uniform(*self.ref_lr_range)
            epochs = np.random.randint(self.ref_epochs_range[0], self.ref_epochs_range[1] + 1)

            ref_net = deepcopy(self.net).to(self.device)
            ref_net.load_state_dict(w_t)
            ref_net.train()
            opt = torch.optim.SGD(ref_net.parameters(), lr=lr, momentum=0.9)
            loader = DataLoader(partition, batch_size=self.batch_size, shuffle=True)

            for _ in range(epochs):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    loss = self.criterion(ref_net(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ref_net.parameters(), self.clip_tau)
                    opt.step()

            refs.append(self._state_delta(ref_net.state_dict(), w_t))
        return refs

    def _compute_local_benign_delta(self, w_t):
        """Δ_loc：本地数据做 1 个 epoch 干净训练"""
        net = deepcopy(self.net).to(self.device)
        net.load_state_dict(w_t)
        net.train()
        opt = torch.optim.SGD(net.parameters(), lr=self.optim_kwargs.get('lr', 0.01), momentum=0.9)
        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            loss = self.criterion(net(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.clip_tau)
            opt.step()
        return self._state_delta(net.state_dict(), w_t)

    # ---------- 锚点 ----------
    def _krum_score_on_refs(self, idx, refs, k):
        """Algorithm 1 line 1：top-k 全部平均"""
        dists = [self._sq_dist_flat(self._flatten(refs[idx]), self._flatten(refs[j]))
                 for j in range(len(refs)) if j != idx]
        topk = torch.topk(torch.stack(dists), min(k, len(dists)), largest=False).values
        return topk.mean().item()

    def _construct_anchor(self, refs, k):
        scores = [self._krum_score_on_refs(i, refs, k) for i in range(len(refs))]
        order = np.argsort(scores)
        d1, d2 = refs[order[0]], refs[order[1]]
        anchor = {n: 0.85 * d1[n] + 0.15 * d2[n] for n in d1}
        return anchor, d1

    # ---------- L_krum（Algorithm 1 line 14）----------
    def _l_krum(self, delta_flat, ref_flats, k):
        dists = torch.stack([self._sq_dist_flat(delta_flat, rf) for rf in ref_flats])
        topk_vals = torch.topk(dists, min(k, len(dists)), largest=False).values
        half_k = max(1, k // 2)
        return topk_vals[:half_k].mean()

    # ---------- 投影 ----------
    def _project(self, delta_adv, delta_loc, refs, delta_1, sigma_ref, median_norm, loc_norm):
        adv_flat = self._flatten(delta_adv)
        d1_flat = self._flatten(delta_1)
        dist_to_d1 = (adv_flat - d1_flat).norm().item()

        if dist_to_d1 > sigma_ref:
            alpha = min(max((dist_to_d1 - sigma_ref) / (dist_to_d1 + 1e-12), 0.0), 0.75)
            delta_adv = {n: (1 - alpha) * delta_adv[n] + alpha * delta_1[n] for n in delta_adv}

        target_norm = max(median_norm, 0.5 * loc_norm)
        cur = self._flatten(delta_adv).norm().item()
        if cur > 1e-12:
            delta_adv = {n: delta_adv[n] * (target_norm / cur) for n in delta_adv}
        return delta_adv

    # ---------- 主流程 ----------
    def train(self, global_weights):
        w_t = global_weights
        eta = self.optim_kwargs.get('lr', 0.01)

        # 1. Δ_loc
        delta_loc = self._compute_local_benign_delta(w_t)
        loc_norm = self._flatten(delta_loc).norm().item()

        # 2. 参考集（每轮重建）
        refs = self._build_references(w_t)
        ref_flats = [self._flatten(r) for r in refs]
        norms = torch.stack([rf.norm() for rf in ref_flats])
        median_norm = norms.median().item()
        mean_norm = norms.mean()
        sigma_ref = torch.stack([(rf.norm() - mean_norm) ** 2 for rf in ref_flats]).mean().sqrt().item()

        # 3. 锚点
        anchor, delta_1 = self._construct_anchor(refs, self.krum_k)
        anchor_flat = self._flatten(anchor)

        # 4. Stage 1：后门注入
        self.net.load_state_dict(w_t)
        self.net.train()
        opt1 = torch.optim.SGD(self.net.parameters(), lr=eta, momentum=0.9)
        for _ in range(self.stage1_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.poison_ratio > 0:
                    x, y = self._poison_batch(x, y)
                opt1.zero_grad()
                loss = self.criterion(self.net(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_tau)
                opt1.step()

        # 5. Stage 2：几何塑造
        opt2 = torch.optim.SGD(self.net.parameters(), lr=0.5 * eta, momentum=0.9)
        loc_norm_t = torch.tensor(loc_norm, device=self.device)

        for _ in range(self.stage2_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.poison_ratio > 0:
                    x, y = self._poison_batch(x, y)

                delta_flat = self._current_delta_flat(w_t)
                l_ce = self.criterion(self.net(x), y)
                l_norm = (delta_flat.norm() - loc_norm_t) ** 2
                l_krum = self._l_krum(delta_flat, ref_flats, self.krum_k)
                l_anchor = ((delta_flat - anchor_flat) ** 2).sum()

                loss = (self.lam_l * l_ce + self.lam_n * l_norm
                        + self.lam_k * l_krum + self.lam_a * l_anchor)
                opt2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_tau)
                opt2.step()

        # 6. 投影
        delta_adv = self._state_delta(self.net.state_dict(), w_t)
        delta_adv = self._project(delta_adv, delta_loc, refs, delta_1,
                                  sigma_ref, median_norm, loc_norm)

        # 7. 写回
        self.net.load_state_dict({k: w_t[k] + delta_adv[k] for k in w_t})

    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.net.state_dict().items()}