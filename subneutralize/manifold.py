"""
================================================================================
   🧠 SubNeutralize v0.2.0: In-Flight Latent Manifold Governor
   Author: Arnab Dutta | Apache-2.0 License
================================================================================
Mathematical Core:
  1. Thin-SVD Prompt Subspace Projection (P_perp = I - V_bias V_bias^T)
  2. Dimensionless Lyapunov Potential (Phi_t = z_erank + z_vel + z_ent - z_conf)
  3. Poincaré Limit-Cycle Orbit Recurrence (rho_orbit = max cos(h_bar_t, h_bar_{t-tau}))
  4. Self-Sizing Roy-Vetterli Coupon-Collector Horizon (r_L1 * ln(r_L2))
================================================================================
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


# ==============================================================================
# 1. ONLINE WELFORD MANIFOLD STATISTICS
# ==============================================================================
class OnlineManifoldStat:
    """
    Numerically stable online computation of mean and variance (Welford's algorithm).
    Computes dynamic z-scores for manifold invariants across the generation trajectory.
    """

    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return math.sqrt(max(1e-12, self.m2 / (self.n - 1)))

    def z_score(self, x: float) -> float:
        return (x - self.mean) / (self.std() + 1e-6)


# ==============================================================================
# 2. PROMPT BIAS SUBSPACE NEUTRALIZER (V_bias)
# ==============================================================================
class PromptSubspaceGovernor:
    """
    Computes Thin-SVD on centered prefill activations at Layer (L-2) to extract V_bias,
    and constructs orthogonal projection operator P_perp = I - V_bias @ V_bias.T.
    Projects out prompt-induced directional bias to isolate pure algorithmic reasoning velocity.
    """

    def __init__(self, rank_k: int = 4):
        self.rank_k = rank_k
        self.V_bias: Optional[torch.Tensor] = None

    def fit_prompt(self, H_prompt: torch.Tensor) -> Tuple[int, int]:
        """
        Fits prompt subspace basis V_bias and computes self-sizing Roy-Vetterli horizons.
        H_prompt: Residual stream activations during prefill [seq_len, hidden_dim].
        Returns: (burn_in_steps, cover_bound_steps)
        """
        mean_h = H_prompt.mean(dim=0, keepdim=True)
        H_centered = (H_prompt - mean_h).float()

        try:
            _, sv, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            k = min(self.rank_k, Vh.shape[0])
            self.V_bias = Vh[:k, :].T.contiguous()

            p1 = sv / (torch.sum(sv) + 1e-12)
            p2 = (sv ** 2) / (torch.sum(sv ** 2) + 1e-12)
            r_l1 = float(torch.exp(-torch.sum(p1 * torch.log(p1 + 1e-12))).item())
            r_l2 = float(torch.exp(-torch.sum(p2 * torch.log(p2 + 1e-12))).item())

            coupon_horizon = int(r_l1 * math.log(max(math.e, r_l2)))
            burn_in = max(180, min(320, int(coupon_horizon * 0.65)))
            cover_bound = max(burn_in + 120, min(460, int(coupon_horizon * 0.95)))
            return burn_in, cover_bound
        except Exception:
            self.V_bias = None
            return 180, 380

    def project(self, h_t: torch.Tensor) -> torch.Tensor:
        """Projects a decoding residual activation: h_perp = h_t - (h_t @ V_bias) @ V_bias.T"""
        if self.V_bias is None:
            return h_t
        return h_t - (h_t @ self.V_bias) @ self.V_bias.T


# ==============================================================================
# 3. UNIFIED GEOMETRIC LOGITS PROCESSOR (HF & vLLM COMPATIBLE)
# ==============================================================================
class SubNeutralizeManifoldProcessor(LogitsProcessor):
    """
    Drop-in Geometric LogitsProcessor for HuggingFace `generate()` and vLLM V1.
    Hooks Decoder Layer (L-2) to monitor the latent reasoning trajectory in real time.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        think_end_token_id: int,
        eos_token_id: int,
        structural_boundary_ids: Optional[Set[int]] = None,
        window_size: int = 16,
        svd_rank: int = 4,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.think_end_token_id = think_end_token_id
        self.eos_token_id = eos_token_id
        self.window_size = window_size
        self.svd_rank = svd_rank

        if structural_boundary_ids is None:
            self.structural_boundary_ids = set()
            for sym in ["\n", ".\n", ":\n", "\n\n", "."]:
                toks = tokenizer.encode(sym, add_special_tokens=False)
                if toks:
                    self.structural_boundary_ids.add(toks[-1])
        else:
            self.structural_boundary_ids = structural_boundary_ids

        self.prompt_governor = PromptSubspaceGovernor(rank_k=svd_rank)
        self.prompt_len: Optional[int] = None
        self.latest_h: Optional[torch.Tensor] = None
        self.prev_h: Optional[torch.Tensor] = None

        self.spectral_burn_in: int = 220
        self.spectral_cover_bound: int = 380

        self.h_window: List[torch.Tensor] = []
        self.centroid_history: List[torch.Tensor] = []
        self.conf_window: List[float] = []

        self.stat_erank = OnlineManifoldStat()
        self.stat_vel = OnlineManifoldStat()
        self.stat_ent = OnlineManifoldStat()
        self.stat_conf = OnlineManifoldStat()

        self.ema_phi: float = 0.0
        self.attractor_steps: int = 0
        self.convergence_ready: bool = False
        self.think_closed: bool = False
        self.think_close_step: Optional[int] = None
        self.triggered_by_manifold: bool = False
        self.telemetry: Dict[str, Any] = {}

        # Attach forward hook to Layer L-2
        layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else (
            model.transformer.h if hasattr(model, "transformer") and hasattr(model.transformer, "h") else None
        )
        if layers is not None:
            target_layer = max(0, len(layers) - 2)
            self.hook_handle = layers[target_layer].register_forward_hook(self._layer_hook)
        else:
            self.hook_handle = None

    def _layer_hook(self, module, inp, outp):
        hidden = outp[0] if isinstance(outp, tuple) else outp
        if hidden.shape[1] > 1:
            # Prefill phase: fit prompt subspace and compute Roy-Vetterli bounds
            H = hidden[0].detach().float()
            burn_in, cover_bound = self.prompt_governor.fit_prompt(H)
            self.spectral_burn_in = burn_in
            self.spectral_cover_bound = cover_bound
            self.latest_h = H[-1:].contiguous()
        else:
            # Autoregressive decoding phase: project out prompt bias
            h_t = hidden[0, -1:, :].detach().float()
            self.latest_h = self.prompt_governor.project(h_t)

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]

        gen_tokens = input_ids[0, self.prompt_len:]
        step = gen_tokens.shape[0]
        if step == 0:
            return scores

        last_tok = int(gen_tokens[-1].item())

        if not self.think_closed and (gen_tokens == self.think_end_token_id).any():
            self.think_closed = True
            if self.think_close_step is None:
                self.think_close_step = step

        # PHASE B: POST-THINK ARTIFACT COMPLETION
        if self.think_closed:
            post_steps = step - (self.think_close_step or step)
            if post_steps >= 60 and last_tok in self.structural_boundary_ids:
                post_ids = gen_tokens[self.think_close_step:]
                post_text = self.tokenizer.decode(post_ids, skip_special_tokens=True)
                blocks = re.findall(r"```(?:python|diff|json|bash)?\n(.*?)```", post_text, re.DOTALL)
                if blocks and len(blocks[-1].strip()) >= 25:
                    scores[:, self.eos_token_id] = scores.max() + 100.0
            return scores

        # PHASE A: LATENT SCRATCHPAD (<think>) GEOMETRIC MANIFOLD MONITORING
        h_t = self.latest_h
        if h_t is None:
            return scores

        if self.prev_h is not None:
            cos_sim = F.cosine_similarity(h_t, self.prev_h, dim=-1).item()
            v_t = max(0.0, 1.0 - cos_sim)
        else:
            v_t = 0.0
        self.prev_h = h_t.clone()

        self.h_window.append(h_t.squeeze(0))
        if len(self.h_window) > self.window_size:
            self.h_window.pop(0)

        erank_t = float(len(self.h_window))
        orbit_recurrence = 0.0
        if len(self.h_window) == self.window_size:
            M = torch.stack(self.h_window, dim=0)
            centroid = F.normalize(M.mean(dim=0, keepdim=True), dim=-1)
            self.centroid_history.append(centroid)
            if len(self.centroid_history) > 96:
                self.centroid_history.pop(0)

            if len(self.centroid_history) > 24:
                lagged = torch.cat(self.centroid_history[:-20], dim=0)
                sims = torch.mm(centroid, lagged.T)
                orbit_recurrence = float(sims.max().item())

            M_c = M - M.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(M_c)
            q = s / (torch.sum(s) + 1e-12)
            erank_t = float(torch.exp(-torch.sum(q * torch.log(q + 1e-12))).item())

        logits_f = scores[0].float()
        probs = F.softmax(logits_f, dim=-1)
        log_probs = F.log_softmax(logits_f, dim=-1)
        h_ent = -torch.sum(probs * log_probs).item()
        max_p = float(torch.max(probs).item())

        self.conf_window.append(max_p)
        if len(self.conf_window) > self.window_size:
            self.conf_window.pop(0)
        deer_conf = float(torch.exp(torch.mean(torch.log(torch.tensor(self.conf_window) + 1e-12))).item())

        if len(self.h_window) == self.window_size:
            if step <= self.spectral_burn_in:
                self.stat_erank.update(erank_t)
                self.stat_vel.update(v_t)
                self.stat_ent.update(h_ent)
                self.stat_conf.update(deer_conf)
            else:
                phi_t = (
                    self.stat_erank.z_score(erank_t)
                    + self.stat_vel.z_score(v_t)
                    + self.stat_ent.z_score(h_ent)
                    - self.stat_conf.z_score(deer_conf)
                )
                self.ema_phi = 0.7 * self.ema_phi + 0.3 * phi_t

                in_attractor = (
                    (self.ema_phi < -1.05)
                    and (deer_conf >= 0.86)
                    and (h_ent < 0.48)
                    and (orbit_recurrence >= 0.84)
                )
                cover_reached = (step >= self.spectral_cover_bound) and (deer_conf >= 0.82)

                if in_attractor or cover_reached:
                    self.attractor_steps += 1
                else:
                    self.attractor_steps = 0
                    self.convergence_ready = False
                    self.stat_erank.update(erank_t)
                    self.stat_vel.update(v_t)
                    self.stat_ent.update(h_ent)
                    self.stat_conf.update(deer_conf)

                if self.attractor_steps >= 4 or cover_reached:
                    self.convergence_ready = True

        if self.convergence_ready and step > 0:
            if last_tok in self.structural_boundary_ids or self.attractor_steps >= 8:
                scores[:, self.think_end_token_id] = scores.max() + 100.0
                self.think_closed = True
                self.think_close_step = step + 1
                self.triggered_by_manifold = True
                self.telemetry = {
                    "step": step,
                    "burn_in": self.spectral_burn_in,
                    "cover_bound": self.spectral_cover_bound,
                    "erank": round(erank_t, 2),
                    "orbit": round(orbit_recurrence, 3),
                    "vel": round(v_t, 4),
                    "ent": round(h_ent, 4),
                    "deer_C": round(deer_conf, 4),
                    "phi": round(self.ema_phi, 2),
                }

        return scores
