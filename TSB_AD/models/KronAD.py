"""KronAD: training-free anomaly detection with a Kronecker Delta Rule memory.

A matrix memory S (per head) stores associations between a context window
(key k_t) and the observation that follows it (value v_t). Keys and values are
L2-normalised random Fourier features, so their overlaps approximate RBF
kernels. Each step reads the association strength y_t = k_t^T S v_t *before*
writing, scores the pair by its error e_t = c - y_t, and then corrects the
association with the Kronecker Delta Rule (Hartl et al., arXiv:2610.26834):

    S <- f * S
    S <- S + beta * (c - k^T S v) * k v^T

With constant gates the decoupled form of the paper (target c / beta) is a
reparameterisation of this coupled form, so only beta and c = 1 are used.

A memory that never forgets absorbs long anomalies: after tens of steps a
sustained level shift reads as familiar and only its onset is flagged. Two
remedies need a train prefix (``fit``):
  ref='max'/'only'  also read a reference copy of S frozen after the train
                    prefix and score max(e, e_ref) per head (or e_ref alone);
  skip_q            do not write pairs whose error exceeds the skip_q quantile
                    of the errors on the second half of the train prefix.
"""
import numpy as np

from ..utils.slidingWindows import find_length_rank


class KronAD:
    def __init__(self, win_mults=(0.5, 1.0, 2.0), n_seg=16, d=64, beta=0.5, tau=None,
                 norm='robust', rule='kron', gate=False, gate_k=3.0, bw_mult=1.0,
                 smooth=False, pred_len=1, local_floor=0.1, ref='none', skip_q=None, chunk=2048,
                 random_state=2024):
        assert norm in ('robust', 'local', 'both')
        assert rule in ('kron', 'delta', 'hebb')
        assert ref in ('none', 'max', 'only')
        self.ref = ref
        self.skip_q = skip_q
        self.S_ref_, self.thr_ = None, None
        self.win_mults = win_mults
        self.n_seg = n_seg
        self.d = d
        self.beta = beta
        self.forget = 1.0 if tau is None else float(np.exp(-1.0 / tau))
        self.norm = norm
        self.rule = rule
        self.gate = gate
        self.gate_k = gate_k
        self.bw_mult = bw_mult
        self.smooth = smooth
        self.pred_len = pred_len
        self.local_floor = local_floor
        self.chunk = chunk
        self.random_state = random_state

    # ---------- preprocessing ----------
    def _fit_scaler(self, X):
        med = np.median(X, axis=0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0)) / 1.349
        std = X.std(axis=0)
        # fall back to std for channels whose IQR collapses (e.g. mostly constant/binary)
        scale = np.where(iqr > 1e-8, iqr, std)
        self.med_, self.scale_ = med, np.maximum(scale, 1e-8)

    def _scale(self, X):
        return (X - self.med_) / self.scale_

    @staticmethod
    def _cumsums(X):
        z = np.zeros((1, X.shape[1]))
        return np.concatenate([z, np.cumsum(X, 0)]), np.concatenate([z, np.cumsum(X ** 2, 0)])

    def _raw_feats(self, X, cs, cs2, t, w, local):
        """Key input (PAA of x[t-w:t]) and value input (x[t:t+p]) for time indices t."""
        n_seg = min(self.n_seg, w)
        edges = np.round(np.linspace(0, w, n_seg + 1)).astype(int)
        lo = (t - w)[:, None] + edges[None, :-1]
        hi = (t - w)[:, None] + edges[None, 1:]
        seg = (cs[hi] - cs[lo]) / (hi - lo)[..., None]                  # (n, n_seg, C)
        val = np.stack([X[t + j] for j in range(self.pred_len)], 1)     # (n, p, C)
        if local:
            mu = (cs[t] - cs[t - w]) / w
            var = (cs2[t] - cs2[t - w]) / w - mu ** 2
            sd = np.maximum(np.sqrt(np.maximum(var, 0)), self.local_floor)
            seg = (seg - mu[:, None]) / sd[:, None]
            val = (val - mu[:, None]) / sd[:, None]
        return seg.reshape(len(t), -1), val.reshape(len(t), -1)

    # ---------- random Fourier features ----------
    def _rff_params(self, Z, rng):
        idx = rng.choice(len(Z), size=min(len(Z), 512), replace=False)
        S = Z[idx]
        sq = (S ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * S @ S.T, 0))
        bw = np.median(D[np.triu_indices(len(S), 1)]) if len(S) > 1 else 1.0
        bw = max(bw, 1e-6) * self.bw_mult
        W = rng.standard_normal((Z.shape[1], self.d)) / bw
        b = rng.uniform(0, 2 * np.pi, self.d)
        return W, b

    @staticmethod
    def _rff(Z, W, b):
        F = np.cos(Z @ W + b)
        return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)

    def _build_heads(self, X_fit):
        base = find_length_rank(X_fit[:, 0].reshape(-1, 1), rank=1)
        base = base if base > 1 else 100
        norms = {'robust': [False], 'local': [True], 'both': [False, True]}[self.norm]
        rng = np.random.default_rng(self.random_state)
        cs, cs2 = self._cumsums(X_fit)
        self.heads_ = []
        for local in norms:
            for m in self.win_mults:
                w = int(np.clip(round(base * m), 4, max(4, len(X_fit) // 4)))
                t = np.arange(w, len(X_fit) - self.pred_len + 1)
                t = rng.choice(t, size=min(len(t), 2048), replace=False)
                Zk, Zv = self._raw_feats(X_fit, cs, cs2, t, w, local)
                self.heads_.append(dict(w=w, local=local, k=self._rff_params(Zk, rng), v=self._rff_params(Zv, rng)))
        self.w_max_ = max(h['w'] for h in self.heads_)
        self.smooth_win_ = base
        H, d = len(self.heads_), self.d
        self.S_ = np.zeros((H, d, d))
        self.n_eff_ = 0.0
        self.e_mu_, self.e_var_, self.n_seen_ = np.zeros(H), np.zeros(H), 0

    # ---------- memory recurrence ----------
    def _run(self, X, score, record=False):
        T = len(X)
        cs, cs2 = self._cumsums(X)
        t_all = np.arange(self.w_max_, T - self.pred_len + 1)
        out = np.full(T, np.nan)
        E = np.full((len(t_all), len(self.heads_)), np.nan) if record else None
        S_ref = self.S_ref_ if score else None
        thr = self.thr_ if score else None
        S, f, beta = self.S_, self.forget, self.beta
        for a in range(0, len(t_all), self.chunk):
            t = t_all[a:a + self.chunk]
            K, V = [], []
            for h in self.heads_:
                Zk, Zv = self._raw_feats(X, cs, cs2, t, h['w'], h['local'])
                K.append(self._rff(Zk, *h['k']))
                V.append(self._rff(Zv, *h['v']))
            K, V = np.stack(K, 1), np.stack(V, 1)  # (n, H, d)
            for i in range(len(t)):
                k, v = K[i], V[i]
                S *= f
                if self.rule == 'delta':
                    r = np.einsum('hk,hkv->hv', k, S)
                    err = v - r
                    e = np.linalg.norm(err, axis=1)
                    u = err
                else:
                    y = np.einsum('hk,hkv,hv->h', k, S, v)
                    if self.rule == 'kron':
                        e = 1.0 - y
                        u = beta * e
                    else:  # hebb: normalise by the effective number of writes
                        self.n_eff_ = f * self.n_eff_ + 1.0
                        e = 1.0 - y / self.n_eff_
                        u = np.ones_like(y)
                if self.gate and self.n_seen_ > self.w_max_:
                    sd = np.sqrt(self.e_var_) + 1e-6
                    g = 1.0 / (1.0 + np.exp(-(self.e_mu_ + self.gate_k * sd - e) / (0.5 * sd)))
                else:
                    g = 1.0
                if thr is not None:  # pairs that look anomalous are not written
                    g = g * (e <= thr)
                u = u * g if self.rule != 'delta' else u * np.reshape(g, (-1, 1))
                if S_ref is not None:
                    if self.rule == 'delta':
                        e_ref = np.linalg.norm(v - np.einsum('hk,hkv->hv', k, S_ref), axis=1)
                    elif self.rule == 'kron':
                        e_ref = 1.0 - np.einsum('hk,hkv,hv->h', k, S_ref, v)
                    else:
                        e_ref = 1.0 - np.einsum('hk,hkv,hv->h', k, S_ref, v) / self.n_eff_ref_
                    e_s = np.maximum(e, e_ref) if self.ref == 'max' else e_ref
                else:
                    e_s = e
                if record:
                    E[a + i] = e
                if self.rule == 'delta':
                    S += beta * k[:, :, None] * u[:, None, :]
                else:
                    S += u[:, None, None] * k[:, :, None] * v[:, None, :]
                # running moments of the error for the contamination gate
                self.n_seen_ += 1
                a_ = max(1.0 / self.n_seen_, 1e-3)
                diff = e - self.e_mu_
                self.e_mu_ = self.e_mu_ + a_ * diff
                self.e_var_ = (1 - a_) * (self.e_var_ + a_ * diff ** 2)
                if score:
                    out[t[i]] = np.maximum(e_s, 0).mean()
        return (out, E) if record else out

    def fit(self, X_train):
        """Warm the memory on (assumed normal) training data without scoring."""
        X_train = np.asarray(X_train, dtype=float)
        self._fit_scaler(X_train)
        Xs = self._scale(X_train)
        self._build_heads(Xs)
        _, E = self._run(Xs, score=False, record=True)
        if self.ref != 'none':
            self.S_ref_, self.n_eff_ref_ = self.S_.copy(), self.n_eff_
        if self.skip_q is not None:
            E = E[len(E) // 2:]  # the memory starts empty: skip the first half of the warm-up
            self.thr_ = np.quantile(E, self.skip_q, axis=0) if len(E) else None
        return self

    def decision_function(self, X):
        X = self._scale(np.asarray(X, dtype=float))
        s = self._run(X, score=True)
        valid = np.flatnonzero(~np.isnan(s))
        if len(valid) == 0:
            return np.zeros(len(X))
        s[:valid[0]] = s[valid[0]]
        s[valid[-1] + 1:] = s[valid[-1]]
        if self.smooth:
            w = self.smooth_win_
            s = np.convolve(np.pad(s, (w // 2, w - 1 - w // 2), mode='edge'), np.ones(w) / w, mode='valid')
        return s

    def fit_predict(self, X):
        """Unsupervised: scaler and bandwidths from the whole series, memory starts empty."""
        X = np.asarray(X, dtype=float)
        self._fit_scaler(X)
        self._build_heads(self._scale(X))
        return self.decision_function(X)
