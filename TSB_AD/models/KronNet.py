"""KronNet: learned key/value encoders for a Kronecker Delta Rule memory.

Variant C of KronAD. A causal TCN maps the context x[..t-1] to keys k_t and an
MLP maps the observation x_t to values v_t (both L2-normalised per head). A
matrix memory per head is updated online with the Kronecker Delta Rule
(Hartl et al., arXiv:2610.26834),

    y_t = k_t^T S v_t ,    S <- S + beta * (1 - y_t) * k_t v_t^T ,

and the pre-write familiarity y_t is the detection signal (score = 1 - y_t).

Training runs the recurrence on segments of the (normal) train prefix with the
memory starting empty, pushing y_t of the true pair towards 1 and y of
synthetic negative pairs (k_t, v(x~_t)) towards 0. The negatives (random other
time, injected spike, rescaling, channel shuffle) prevent collapse of the
encoders to constants.
"""
import math

import numpy as np
import torch
from torch import nn

from ..utils.slidingWindows import find_length_rank
from ..utils.torch_utility import get_gpu


class CausalTCN(nn.Module):
    def __init__(self, n_feats, hidden, n_layers):
        super().__init__()
        self.inp = nn.Conv1d(n_feats, hidden, 1)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=2 ** i) for i in range(n_layers)])
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, C, T) -> (B, hidden, T), causal
        h = self.inp(x)
        for conv in self.convs:
            pad = 2 * conv.dilation[0]
            h = h + self.act(conv(nn.functional.pad(h, (pad, 0))))
        return h


class KronNetModel(nn.Module):
    def __init__(self, n_feats, d=32, heads=4, hidden=64, n_layers=5, beta_init=0.1, rff=None, key_rff=None,
                 tcn=True):
        super().__init__()
        self.d, self.heads = d, heads
        if tcn:
            self.tcn = CausalTCN(n_feats, hidden, n_layers)
            self.key = nn.Linear(hidden, heads * d)
        else:
            self.tcn = None
        # fixed KronAD keys: RFF of the PAA of x[t-w_h:t], one window length per head
        self.key_rff = key_rff
        if key_rff is not None:
            for h, (w, edges, W, b) in enumerate(key_rff):
                self.register_buffer(f'kW{h}', W)
                self.register_buffer(f'kb{h}', b)
                self.register_buffer(f'kedges{h}', edges)
            self.key_w = [w for w, *_ in key_rff]
            # zero-initialised residual weight of the learned keys: at init the keys equal KronAD's
            self.key_alpha = nn.Parameter(torch.zeros(heads))
        if rff is None:
            self.value = nn.Sequential(nn.Linear(n_feats, hidden), nn.GELU(), nn.Linear(hidden, heads * d))
        else:  # fixed random Fourier features of x_t (as in KronAD); only the keys are learned
            W, b = rff
            self.register_buffer('rff_W', W)
            self.register_buffer('rff_b', b)
            self.value = lambda x: torch.cos(x @ self.rff_W + self.rff_b)
        self.beta_logit = nn.Parameter(torch.full((heads,), math.log(beta_init / (1 - beta_init))))

    def keys(self, x):  # x: (B, T, C) -> (B, T, H, d); k_t only sees x[..t-1]
        k = None
        if self.tcn is not None:
            h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
            h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
            k = nn.functional.normalize(self.key(h).unflatten(-1, (self.heads, self.d)), dim=-1)
        if self.key_rff is None:
            return k
        kr = self.rff_keys(x)
        if k is None:
            return kr
        return nn.functional.normalize(kr + self.key_alpha[:, None] * k, dim=-1)

    def rff_keys(self, x):
        B, T, C = x.shape
        cs = torch.cat([x.new_zeros(B, 1, C), torch.cumsum(x, 1)], 1)  # cs[t] = sum x[:t]
        t = torch.arange(T, device=x.device)
        out = []
        for h, w in enumerate(self.key_w):
            edges = getattr(self, f'kedges{h}')
            lo = (t[:, None] - w + edges[None, :-1]).clamp(min=0)
            hi = (t[:, None] - w + edges[None, 1:]).clamp(min=0)
            seg = (cs[:, hi] - cs[:, lo]) / (hi - lo).clamp(min=1)[None, ..., None]  # (B, T, n_seg, C)
            f = torch.cos(seg.flatten(2) @ getattr(self, f'kW{h}') + getattr(self, f'kb{h}'))
            out.append(nn.functional.normalize(f, dim=-1))
        return torch.stack(out, 2)

    def values(self, x):  # x: (..., C) -> (..., H, d)
        v = self.value(x).unflatten(-1, (self.heads, self.d))
        return nn.functional.normalize(v, dim=-1)

    def recur(self, k, v, v_neg=None, S=None, chunk=256):
        """k, v: (B, T, H, d); v_neg: (B, T, N, H, d). Returns pre-write y (B, T, H), y_neg (B, T, N, H), S.

        Chunkwise-parallel and exact: every write is u_t * k_t v_t^T with scalar u_t = beta * (1 - y_t), and
        y_t = y0_t + sum_{i<t} u_i (k_t.k_i)(v_t.v_i), so u solves (I + beta * tril(KK^T * VV^T, -1)) u = beta * (1 - y0).
        """
        B, T, H, d = k.shape
        S = k.new_zeros(B, H, d, d) if S is None else S
        beta = torch.sigmoid(self.beta_logit)[None, :, None]
        ys, ys_neg = [], []
        for a in range(0, T, chunk):
            kc, vc = k[:, a:a + chunk].transpose(1, 2), v[:, a:a + chunk].transpose(1, 2)  # (B, H, n, d)
            n = kc.shape[2]
            Kg = torch.tril(kc @ kc.transpose(-1, -2), -1)
            A = Kg * (vc @ vc.transpose(-1, -2))
            Sk = kc @ S                                                              # rows k_t^T S
            y0 = (Sk * vc).sum(-1)                                                   # (B, H, n)
            M = torch.eye(n, device=k.device) + beta[..., None] * A
            u = torch.linalg.solve_triangular(M, (beta * (1 - y0))[..., None], upper=False, unitriangular=True)[..., 0]
            ys.append((y0 + (A @ u[..., None])[..., 0]).transpose(1, 2))
            if v_neg is not None:
                vn = v_neg[:, a:a + chunk].permute(0, 3, 1, 2, 4)                    # (B, H, n, N, d)
                yn = torch.einsum('bhtv,bhtmv->bhtm', Sk, vn)
                yn = yn + torch.einsum('bhti,bhtmv,bhiv,bhi->bhtm', Kg, vn, vc, u)
                ys_neg.append(yn.permute(0, 2, 3, 1))
            S = S + torch.einsum('bhi,bhik,bhiv->bhkv', u, kc, vc)
        y = torch.cat(ys, 1)
        y_neg = torch.cat(ys_neg, 1) if v_neg is not None else None
        return y, y_neg, S


def make_negatives(x, rng_gen):
    """x: (B, T, C) normalised. Returns (B, T, 4, C) negative observations."""
    B, T, C = x.shape
    idx = torch.randint(0, T, (B, T), device=x.device, generator=rng_gen)
    rand_time = torch.gather(x, 1, idx[..., None].expand(-1, -1, C))
    mask = (torch.rand(B, T, C, device=x.device, generator=rng_gen) < max(1.0 / C, 0.3)).float()
    sign = torch.randint(0, 2, (B, T, C), device=x.device, generator=rng_gen) * 2 - 1
    mag = 1 + 3 * torch.rand(B, T, C, device=x.device, generator=rng_gen)
    spike = x + mask * sign * mag
    s = torch.where(torch.rand(B, T, 1, device=x.device, generator=rng_gen) < 0.5,
                    0.5 + 0.3 * torch.rand(B, T, 1, device=x.device, generator=rng_gen),
                    1.2 + 0.8 * torch.rand(B, T, 1, device=x.device, generator=rng_gen))
    scale = x * s + (1 - s) * mask * sign  # also shifts near-zero values
    if C > 1:
        shuffle = x[..., torch.randperm(C, device=x.device, generator=rng_gen)]
    else:
        idx2 = torch.randint(0, T, (B, T), device=x.device, generator=rng_gen)
        shuffle = torch.gather(x, 1, idx2[..., None].expand(-1, -1, C))
    return torch.stack([rand_time, spike, scale, shuffle], 2)


class KronNet:
    def __init__(self, window_size=64, d=32, heads=4, hidden=64, beta_init=0.1, lr=1e-3,
                 n_iters=300, batch_size=32, seg_len=256, neg_weight=1.0, values='mlp',
                 bw_mults=(0.5, 1.0, 2.0, 4.0), keys='tcn', win_mults=(0.5, 1.0, 2.0), n_seg=16,
                 random_state=2024):
        assert values in ('mlp', 'rff')
        assert keys in ('tcn', 'rff', 'rff+tcn')
        self.values, self.bw_mults = values, bw_mults
        self.keys, self.win_mults, self.n_seg = keys, win_mults, n_seg
        if keys != 'tcn':
            heads = len(win_mults)
        self.n_layers = max(1, int(math.floor(math.log2((window_size - 1) / 2 + 1))))
        self.rf = 1 + 2 * (2 ** self.n_layers - 1)
        self.d, self.heads, self.hidden, self.beta_init = d, heads, hidden, beta_init
        self.lr, self.n_iters, self.batch_size, self.seg_len = lr, n_iters, batch_size, seg_len
        self.neg_weight = neg_weight
        self.random_state = random_state
        self.device = get_gpu(True)

    def _fit_scaler(self, X):
        med = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0)) / 1.349
        scale = np.where(iqr > 1e-8, iqr, X.std(axis=0))
        self.med_, self.scale_ = med, np.maximum(scale, 1e-8)

    def _scale(self, X):
        return torch.tensor((X - self.med_) / self.scale_, dtype=torch.float32, device=self.device)

    def _rff_params(self, X, gen):
        """One RFF map per head, bandwidth = median pairwise distance of x_t times bw_mults[h % len]."""
        Z = X[torch.randperm(len(X), device=X.device, generator=gen)[:512]]
        D = torch.cdist(Z, Z)
        bw = max(D[torch.triu(torch.ones_like(D), 1) > 0].median().item(), 1e-6) if len(Z) > 1 else 1.0
        bws = torch.tensor([bw * self.bw_mults[h % len(self.bw_mults)] for h in range(self.heads)], device=X.device)
        W = torch.randn(X.shape[1], self.heads, self.d, device=X.device, generator=gen) / bws[:, None]
        b = 2 * math.pi * torch.rand(self.heads, self.d, device=X.device, generator=gen)
        return W.flatten(1), b.flatten()

    @staticmethod
    def _median_bw(Z):
        D = torch.cdist(Z, Z)
        return max(D[torch.triu(torch.ones_like(D), 1) > 0].median().item(), 1e-6) if len(Z) > 1 else 1.0

    def _key_rff_params(self, X, X_raw, gen):
        """KronAD keys: window lengths from the dominant period, bandwidth by the median heuristic."""
        base = find_length_rank(X_raw[:, 0].reshape(-1, 1), rank=1)
        base = base if base > 1 else 100
        T, C = X.shape
        cs = torch.cat([X.new_zeros(1, C), torch.cumsum(X, 0)])
        params = []
        for m in self.win_mults:
            w = int(np.clip(round(base * m), 4, max(4, T // 4)))
            n_seg = min(self.n_seg, w)
            edges = torch.tensor(np.round(np.linspace(0, w, n_seg + 1)).astype(int), device=X.device)
            t = torch.arange(w, T, device=X.device)
            t = t[torch.randperm(len(t), device=X.device, generator=gen)[:512]]
            lo, hi = t[:, None] - w + edges[None, :-1], t[:, None] - w + edges[None, 1:]
            Z = ((cs[hi] - cs[lo]) / (hi - lo)[..., None]).flatten(1)
            W = torch.randn(Z.shape[1], self.d, device=X.device, generator=gen) / self._median_bw(Z)
            b = 2 * math.pi * torch.rand(self.d, device=X.device, generator=gen)
            params.append((w, edges, W, b))
        return params

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float)
        torch.manual_seed(self.random_state)
        gen = torch.Generator(device=self.device).manual_seed(self.random_state)
        self._fit_scaler(X_train)
        X = self._scale(X_train)
        T, C = X.shape
        rff = self._rff_params(X, gen) if self.values == 'rff' else None
        key_rff = self._key_rff_params(X, X_train, gen) if self.keys != 'tcn' else None
        self.model = KronNetModel(C, self.d, self.heads, self.hidden, self.n_layers, self.beta_init, rff, key_rff,
                                  tcn=self.keys != 'rff').to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # context the keys need before the first scored step
        self.ctx = max(self.rf if self.keys != 'rff' else 0, max((p[0] for p in key_rff or []), default=0))
        # each sample: ctx steps of key context + L steps of recurrence
        L = int(min(self.seg_len, max(16, T - self.ctx - 1)))
        burn = L // 4
        span = self.ctx + L
        self.model.train()
        for it in range(self.n_iters):
            starts = torch.randint(0, max(1, T - span + 1), (self.batch_size,), device=self.device, generator=gen)
            seg = torch.stack([X[s:s + span] for s in starts.tolist()])
            if seg.shape[1] < span:  # train prefix shorter than one span
                seg = nn.functional.pad(seg.transpose(1, 2), (span - seg.shape[1], 0), mode='replicate').transpose(1, 2)
            k = self.model.keys(seg)[:, self.ctx:]
            x_t = seg[:, self.ctx:]
            v = self.model.values(x_t)
            v_neg = self.model.values(make_negatives(x_t, gen))
            y, y_neg, _ = self.model.recur(k, v, v_neg)
            loss = ((1 - y[:, burn:]) ** 2).mean() + self.neg_weight * (y_neg[:, burn:] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
        self.train_loss_ = loss.item()
        self.X_train_ = X_train
        return self

    @torch.no_grad()
    def _stream(self, X, S=None, chunk=4096):
        """Run the memory over a whole series; returns per-step familiarity y (T,) and final S."""
        Xs = self._scale(X)[None]
        ys = []
        for a in range(0, Xs.shape[1], chunk):
            # keys are causal with a receptive field <= ctx, so a ctx-long prefix makes chunked keys exact
            a0 = max(0, a - self.ctx)
            k = self.model.keys(Xs[:, a0:a + chunk])[:, a - a0:]
            v = self.model.values(Xs[:, a:a + chunk])
            y, _, S = self.model.recur(k, v, S=S)
            ys.append(y[0].mean(-1))
        return torch.cat(ys).cpu().numpy(), S

    def decision_function(self, X):
        self.model.eval()
        _, S = self._stream(self.X_train_)  # warm the memory on the train prefix
        y, _ = self._stream(np.asarray(X, dtype=float), S=S)
        score = 1.0 - y
        c = min(self.ctx, len(score) - 1)
        score[:c] = score[c]
        return score
