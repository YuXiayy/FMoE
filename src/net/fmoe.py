"""Frequency-structured mixture-of-experts restoration network."""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numbers

from einops import rearrange
from fvcore.nn import FlopCountAnalysis

from net.band_sequence import BandSequenceExpert
from net.degradation_encoder import CompoundDegradationEncoder


##########################################################################
def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax with temperature to the logits.
    
    Args:
    - logits (torch.Tensor): The input logits.
    - temperature (float): The temperature factor.
    
    Returns:
    - torch.Tensor: The softmax output with temperature.
    """
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    return F.softmax(scaled_logits, dim=-1)

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = stitched.new_zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            expert_out[-1].size(2),
            expert_out[-1].size(3),
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined
    
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x 
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, y):
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
##########################################################################
## Frequency-structured expert adapter
class SharedConditionedBandExpert(nn.Module):
    """Adapt a latent BandSequenceExpert to the existing shared-expert API.

    Inputs and output are ``[B,C,F,T]``.  The band expert predicts a residual;
    a learned shared-feature gate preserves the original decoder's shared
    expert conditioning.  Empty sparse-dispatch batches are returned without
    entering attention kernels.
    """

    def __init__(self, channels: int, **band_config):
        super().__init__()
        self.body = BandSequenceExpert(channels=channels, **band_config)
        self.shared_gate = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        delta = self.body(x) - x
        return x + delta * torch.sigmoid(self.shared_gate(shared))
        



########################################################################### 
## Adapter Layer
class AdapterLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        condition_dim: int,
        band_expert_config: Dict,
        routing_balance_ema_decay: float = 0.99,
        enable_routing_logging: bool = False,
        profile_expert_flops: bool = False,
    ):
        super().__init__()            
        
        self.loss = None
        self.num_experts = num_experts

        config = dict(band_expert_config)
        freq_bins = int(config.pop("freq_bins"))
        embed_dims = list(config.pop("embed_dims"))
        band_counts = list(config.pop("num_bands"))
        block_depths = list(config.pop("depths"))
        head_counts = list(config.pop("num_heads"))
        lengths = {len(embed_dims), len(band_counts), len(block_depths), len(head_counts)}
        if lengths != {num_experts}:
            raise ValueError(
                "Band expert configuration lists must all match num_experts="
                f"{num_experts}."
            )
        self.experts = nn.ModuleList(
            [
                SharedConditionedBandExpert(
                    channels=dim,
                    freq_bins=freq_bins,
                    embed_dim=embed_dims[index],
                    num_bands=band_counts[index],
                    num_blocks=block_depths[index],
                    num_heads=head_counts[index],
                    **config,
                )
                for index in range(num_experts)
            ]
        )
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.profile_expert_flops = bool(profile_expert_flops)
        self._flops_profiled = False
        self.flop_profile_metadata: Dict[str, object] = {}
        self.register_buffer(
            "expert_flops",
            torch.full((num_experts,), float("nan"), dtype=torch.float64),
            persistent=False,
        )
        self.routing = RoutingFunction(
            condition_dim,
            num_experts=num_experts,
            complexity=expert_complexity,
            balance_ema_decay=routing_balance_ema_decay,
            enable_routing_logging=enable_routing_logging,
        )

    def _profile_expert_flops_once(self, x, shared):
        """Profile one real stage shape on rank zero using fvcore tracing."""

        if self._flops_profiled or not self.profile_expert_flops:
            return
        self._flops_profiled = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        sample_x = x[:1].detach()
        sample_shared = shared[:1].detach()
        values = []
        unsupported = []
        for expert_index, expert in enumerate(self.experts):
            was_training = expert.training
            try:
                expert.eval()
                with torch.no_grad():
                    analysis = FlopCountAnalysis(expert, (sample_x, sample_shared))
                    analysis.unsupported_ops_warnings(False)
                    analysis.uncalled_modules_warnings(False)
                    analysis.tracer_warnings("none")
                    values.append(float(analysis.total()))
                    unsupported.append(
                        {
                            str(name): int(count)
                            for name, count in analysis.unsupported_ops().items()
                        }
                    )
            except Exception as error:
                raise RuntimeError(
                    f"Failed to profile routing expert {expert_index} at "
                    f"shape {tuple(sample_x.shape)}"
                ) from error
            finally:
                expert.train(was_training)
        self.expert_flops.copy_(
            torch.tensor(values, device=self.expert_flops.device, dtype=torch.float64)
        )
        self.flop_profile_metadata = {
            "method": "fvcore.FlopCountAnalysis",
            "input_shape": list(sample_x.shape),
            "expert_flops": values,
            "unsupported_ops": unsupported,
        }
        
    def forward(self, x, z_deg, shared, stage_bias=None):
        self._profile_expert_flops_once(x, shared)
        gates, aux_loss = self.routing(z_deg, stage_bias=stage_bias)
        self.loss = aux_loss
        if self.routing.last_routing_stats and bool(
            torch.isfinite(self.expert_flops).all()
        ):
            self.routing.last_routing_stats["expert_flops"] = (
                self.expert_flops.detach()
            )
                
        # The same sparse dispatcher is used for training and inference.  In
        # eval mode the router is deterministic, and this path remains correct
        # for arbitrary batch sizes and per-sample expert selections.
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_shared_inputs = dispatcher.dispatch(shared)
        expert_outputs = [
            self.experts[index](expert_inputs[index], expert_shared_inputs[index])
            for index in range(len(self.experts))
        ]
        out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
            
        out = self.proj_out(out)
        return out

    

class RoutingFunction(nn.Module):
    """Project the global audio-derived degradation condition to expert logits.

    Each routed decoder block owns an independent condition projection.  The
    stage bias is supplied by its DecoderResidualGroup and is therefore shared
    by all routed blocks within that decoder stage.
    """

    def __init__(
        self,
        condition_dim,
        num_experts,
        complexity,
        balance_ema_decay=0.99,
        enable_routing_logging=False,
    ):
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if not 0.0 < balance_ema_decay < 1.0:
            raise ValueError("routing balance EMA decay must be in (0, 1)")

        self.condition_dim = int(condition_dim)
        self.condition_projection = nn.Linear(
            self.condition_dim, num_experts, bias=False
        )
        complexity = complexity.to(dtype=torch.float32)
        self.register_buffer('complexity', complexity.clone())
        complexity_bias = complexity / complexity.min()
        self.register_buffer('complexity_bias', complexity_bias)
        
        self.k = 1
        self.tau = 1.0
        self.num_experts = num_experts
        self.noise_std = 1.0 / num_experts
        self.balance_ema_decay = float(balance_ema_decay)
        importance_target = self.complexity_bias.clamp_min(1.0e-8).reciprocal()
        importance_target = importance_target / importance_target.sum()
        load_target = torch.full_like(
            self.complexity_bias, 1.0 / self.num_experts
        )
        self.register_buffer("importance_ema", importance_target)
        self.register_buffer("load_ema", load_target)
        self.enable_routing_logging = bool(enable_routing_logging)
        self.last_routing_stats: Dict[str, torch.Tensor] = {}

    @property
    def degradation_gate(self):
        """Compatibility-friendly semantic alias for the global condition projection."""

        return self.condition_projection

    def forward(self, z_deg, stage_bias=None):
        if z_deg.ndim != 2 or z_deg.shape[1] != self.condition_dim:
            raise ValueError(
                "z_deg must have shape [batch, {}], got {}".format(
                    self.condition_dim, tuple(z_deg.shape)
                )
            )
        logits = self.condition_projection(z_deg)
        if stage_bias is None or stage_bias.shape != (self.num_experts,):
            raise ValueError("stage_bias must have shape [num_experts]")
        logits = logits + stage_bias
        if self.training:
            loss_imp = self.importance_loss(
                softmax_with_temperature(logits, self.tau)
            )
        
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        gating_scores = softmax_with_temperature(noisy_logits, self.tau)
        if not bool(torch.isfinite(gating_scores).all()):
            raise FloatingPointError("routing probabilities contain NaN or Inf")
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        # Final auxiliary loss
        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = logits.new_zeros(())
        
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        if self.enable_routing_logging:
            self.last_routing_stats = {
                # Dense pre-top-k probabilities retain meaningful entropy even
                # when the execution path dispatches only one expert.
                "gates": gating_scores.detach(),
                "topk_indices": top_k_indices.detach(),
                "load": F.one_hot(
                    top_k_indices, num_classes=self.num_experts
                ).sum(dim=(0, 1)).to(dtype=gating_scores.dtype).detach(),
                "importance": gating_scores.sum(dim=0).detach(),
                "expert_parameters": self.complexity.detach(),
            }
        else:
            self.last_routing_stats = {}
        return gates, aux_loss

    def importance_loss(self, gating_scores):
        return self._ema_balance_loss(
            gating_scores,
            self.importance_ema,
            apply_complexity=True,
        )

    @staticmethod
    def _coefficient_of_variation_squared(values):
        mean = values.mean()
        std = values.std() if values.numel() > 1 else values.new_zeros(())
        return (std / (mean + 1e-8)) ** 2

    @staticmethod
    def _distributed_detached_mean(values):
        mean = values.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.SUM)
            mean /= torch.distributed.get_world_size()
        return mean

    def _ema_balance_loss(self, per_sample_values, running_ema, *, apply_complexity):
        """Balance a rolling utilization estimate without retaining old graphs.

        The previous EMA is detached.  Only the current micro-batch contributes
        gradients, scaled by ``1 - decay``; the persistent statistic is updated
        from the detached all-rank mean after the loss expression is formed.
        This avoids the upstream batch estimator degenerating into a per-sample
        uniformity penalty when the physical batch size is one.
        """

        if running_ema is None:
            raise RuntimeError("EMA balance mode requires an initialized buffer")
        current_mean = per_sample_values.mean(dim=0)
        decay = self.balance_ema_decay
        blended = running_ema.detach() * decay + current_mean * (1.0 - decay)
        if apply_complexity:
            blended = blended * self.complexity_bias
        loss = self._coefficient_of_variation_squared(blended)

        global_current = self._distributed_detached_mean(current_mean)
        with torch.no_grad():
            running_ema.mul_(decay).add_(global_current, alpha=1.0 - decay)
        return loss

    def _probability_in_top_k(self, logits, logits_noisy, noise_std):
        """Differentiable noisy top-k membership probability.

        Selected experts must clear the ``(k + 1)``-th noisy logit to remain
        selected, while unselected experts must clear the ``k``-th logit to
        enter.  Using one shared selected-expert threshold, as the inherited
        implementation did, biases the load estimate and gives the boundary
        expert a self-referential threshold.
        """

        if logits.shape != logits_noisy.shape or logits.ndim != 2:
            raise ValueError("routing logits must have matching [batch, experts] shape")
        if logits.shape[-1] != self.num_experts:
            raise ValueError("routing logits expert dimension does not match router")
        if noise_std <= 0:
            raise ValueError("noisy top-k probability requires positive noise_std")
        if self.k >= self.num_experts:
            return torch.ones_like(logits)

        top_values = torch.topk(logits_noisy, self.k + 1, dim=-1).values
        threshold_if_in = top_values[:, self.k : self.k + 1]
        threshold_if_out = top_values[:, self.k - 1 : self.k]
        is_in = logits_noisy > threshold_if_in
        scale = float(noise_std) * math.sqrt(2.0)
        probability_if_in = 0.5 * (
            1.0 + torch.erf((logits - threshold_if_in) / scale)
        )
        probability_if_out = 0.5 * (
            1.0 + torch.erf((logits - threshold_if_out) / scale)
        )
        return torch.where(is_in, probability_if_in, probability_if_out)

    def load_loss(self, logits, logits_noisy, noise_std):
        p = self._probability_in_top_k(logits, logits_noisy, noise_std)

        return self._ema_balance_loss(
            p,
            self.load_ema,
            apply_complexity=False,
        )




##########################################################################
## Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        
        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type)
        ])
        
        self.mixer = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x
        


##########################################################################
## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        num_experts,
        condition_dim,
        band_expert_config,
        routing_balance_ema_decay=0.99,
        enable_routing_logging=False,
        profile_expert_flops=False,
    ):
        super().__init__()

        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type),
        ])
        
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])
        
        self.shared = Attention(dim, num_heads, bias)
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.adapter = AdapterLayer(
            dim=dim,
            num_experts=num_experts,
            condition_dim=condition_dim,
            band_expert_config=band_expert_config,
            routing_balance_ema_decay=routing_balance_ema_decay,
            enable_routing_logging=enable_routing_logging,
            profile_expert_flops=profile_expert_flops,
        )
        
    def forward(self, x, z_deg, stage_bias=None):
        shortcut = x
        x = self.norms[0](x)
        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter(x_a, z_deg, x_s, stage_bias=stage_bias)
        loss = self.adapter.loss
        x = self.mixer(x_a, x_s) + shortcut

        x = x + self.ffn(self.norms[1](x))
        return x, loss

    

######################################################################
## Encoder Residual Group
class EncoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
            )

    def forward(self, x):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x)
            i += 1
        return x    
    
    
    
######################################################################
## Decoder Residual Group
class DecoderResidualGroup(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: List[int],
        num_blocks: int,
        ffn_expansion: int,
        LayerNorm_type: str,
        bias: bool,
        num_experts: int,
        condition_dim: int,
        band_expert_config: Dict,
        routing_balance_ema_decay: float = 0.99,
        enable_routing_logging: bool = False,
        profile_expert_flops: bool = False,
    ):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        self.enable_routing_logging = bool(enable_routing_logging)
        self.routing_stats: Dict[str, torch.Tensor] = {}
        self.stage_bias = nn.Parameter(torch.zeros(num_experts))
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    dim,
                    num_heads,
                    ffn_expansion,
                    bias,
                    LayerNorm_type,
                    num_experts=num_experts,
                    condition_dim=condition_dim,
                    band_expert_config=band_expert_config,
                    routing_balance_ema_decay=routing_balance_ema_decay,
                    enable_routing_logging=enable_routing_logging,
                    profile_expert_flops=profile_expert_flops and i == 0,
                )
            )

    def forward(self, x, z_deg):
        i = 0
        self.loss = 0
        block_statistics = []
        while i < len(self.layers):
            x , loss = self.layers[i](x, z_deg, stage_bias=self.stage_bias)
            self.loss += loss
            if self.enable_routing_logging and self.layers[i].adapter is not None:
                statistics = self.layers[i].adapter.routing.last_routing_stats
                if statistics:
                    block_statistics.append(statistics)
            i += 1
        if block_statistics:
            self.routing_stats = {
                "gates": torch.stack(
                    [statistics["gates"] for statistics in block_statistics],
                    dim=1,
                ),
                "topk_indices": torch.stack(
                    [statistics["topk_indices"] for statistics in block_statistics],
                    dim=1,
                ),
                "load": torch.stack(
                    [statistics["load"] for statistics in block_statistics]
                ).sum(dim=0),
                "importance": torch.stack(
                    [statistics["importance"] for statistics in block_statistics]
                ).sum(dim=0),
                # Complexity is per routing decision.  Blocks in a decoder
                # stage share the same expert architecture, so averaging avoids
                # double-counting when selection counts already include blocks.
                "expert_parameters": torch.stack(
                    [
                        statistics["expert_parameters"]
                        for statistics in block_statistics
                    ]
                ).mean(dim=0),
            }
            profiled_flops = [
                statistics["expert_flops"]
                for statistics in block_statistics
                if "expert_flops" in statistics
            ]
            if profiled_flops:
                self.routing_stats["expert_flops"] = torch.stack(
                    profiled_flops
                ).mean(dim=0)
            for layer in self.layers:
                if layer.adapter is not None:
                    layer.adapter.routing.last_routing_stats = {}
        else:
            self.routing_stats = {}
        return x  
    
    
     
##########################################################################
## Overlapped spectrogram patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
    
    
##########################################################################
##
class FMoE(nn.Module):
    """F-MoE architecture released with the paper."""

    def __init__(
        self,
        inp_channels=4,
        out_channels=4,
        dim=48,
        levels: int = 4,
        heads=(1, 2, 4, 8),
        num_blocks=(4, 6, 6, 8),
        num_dec_blocks=(2, 4, 4),
        ffn_expansion_factor=2,
        num_refinement_blocks=4,
        LayerNorm_type="WithBias",
        bias=False,
        num_experts=4,
        input_freq_bins=520,
        expert_embed_dims=(48, 64, 96, 128),
        expert_num_bands=(16, 24, 32, 48),
        expert_depths=(1, 2, 3, 4),
        expert_num_heads=(2, 4, 4, 8),
        sample_rate=44100,
        n_fft=1024,
        degradation_embed_dim=128,
        num_degradation_classes=19,
        include_training_head=False,
        routing_balance_ema_decay=0.99,
        enable_routing_logging=False,
        profile_expert_flops=False,
    ):
        super().__init__()
        if inp_channels != 4 or out_channels != 4:
            raise ValueError("F-MoE requires stereo complex input and output channels")
        if levels != 4 or num_experts != 4:
            raise ValueError("the released F-MoE architecture uses four levels and four experts")
        if profile_expert_flops and not enable_routing_logging:
            raise ValueError("expert FLOP profiling requires routing logging")
        downsample_factor = 2 ** (levels - 1)
        if input_freq_bins % downsample_factor:
            raise ValueError(
                f"input_freq_bins={input_freq_bins} must be divisible by "
                f"the encoder factor {downsample_factor}"
            )

        self.levels = levels
        self.num_blocks = list(num_blocks)
        self.num_dec_blocks = list(num_dec_blocks)
        self.num_refinement_blocks = num_refinement_blocks
        self.enable_routing_logging = bool(enable_routing_logging)
        decoder_freq_bins = [
            input_freq_bins // (2 ** (levels - 2 - index))
            for index in range(levels - 1)
        ]
        expert_lists = {
            len(expert_embed_dims),
            len(expert_num_bands),
            len(expert_depths),
            len(expert_num_heads),
        }
        if expert_lists != {num_experts}:
            raise ValueError("all F-MoE expert lists must contain four entries")

        dims = [dim * 2**index for index in range(levels)]
        self.patch_embed = OverlapPatchEmbed(
            in_c=inp_channels, embed_dim=dim, bias=False
        )
        self.degradation_encoder = CompoundDegradationEncoder(
            degradation_embed_dim=degradation_embed_dim,
            num_degradation_classes=num_degradation_classes,
            include_training_head=include_training_head,
        )
        routing_condition_dim = degradation_embed_dim
                
        # -- Encoder --        
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i], 
                    num_blocks=num_blocks[i], 
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=True,),
                Downsample(dim*2**i)
                ])
            )
        
        # -- Latent --
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1], 
            num_heads=heads[-1], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
                  
        # -- Decoder --
        dims = dims[::-1]
        heads = list(heads)[::-1]
        decoder_blocks = list(num_dec_blocks)[::-1]
        
        self.dec = nn.ModuleList([])
        for i in range(levels-1):
            band_expert_config = {
                "freq_bins": decoder_freq_bins[i],
                "embed_dims": expert_embed_dims,
                "num_bands": expert_num_bands,
                "depths": expert_depths,
                "num_heads": expert_num_heads,
                "sample_rate": sample_rate,
                "n_fft": n_fft,
            }
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i+1],
                    num_blocks=decoder_blocks[i],
                    num_heads=heads[i+1],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type,
                    bias=bias,
                    num_experts=num_experts,
                    condition_dim=routing_condition_dim,
                    band_expert_config=band_expert_config,
                    routing_balance_ema_decay=routing_balance_ema_decay,
                    enable_routing_logging=self.enable_routing_logging,
                    profile_expert_flops=profile_expert_flops),
                ])
            )

        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks, 
            num_heads=heads[0], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
        
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None
    
    def forward(self, x):
        model_input = x
        z_deg, degradation_logits = self.degradation_encoder(model_input)

        feats = self.patch_embed(x)
        
        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)
        
        feats = self.latent(feats)
                
        routing_stats = {}
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, z_deg)
            self.total_loss += block.loss
            if self.enable_routing_logging and block.routing_stats:
                routing_stats[f"decoder_stage_{i}"] = block.routing_stats
                block.routing_stats = {}

        feats = self.refinement(feats)
        prediction = self.output(feats)
        restored = prediction + model_input

        self.total_loss /= sum(self.num_dec_blocks)
        return {
            "restored_spec": restored,
            "degradation_embedding": z_deg,
            "degradation_logits": degradation_logits,
            "balance_loss": self.total_loss,
            "routing_stats": routing_stats,
        }
