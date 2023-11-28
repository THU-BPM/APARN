import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from inspect import isfunction
from functools import partial
from dataclasses import dataclass
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# from alphafold2_pytorch.utils import *


# constants

@dataclass
class Recyclables:
    coords: torch.Tensor
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor


@dataclass
class ReturnValues:
    distance: torch.Tensor = None
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None


# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


# helper classes

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val


# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)


# attention

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            seq_len=None,
            heads=8,
            dim_head=64,
            dropout=0.,
            gating=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim), (q, k))
            q = q.mean(dim=1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device=device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2],
                                                                                                 device=device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        # gates = self.gating(x)
        # out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            row_attn=True,
            col_attn=True,
            accept_edges=False,
            global_query_attn=False,
            edge_dim=None,
            **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim=dim, heads=heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(default(edge_dim, dim), heads, bias=False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges=None, mask=None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x=axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim)
        out = rearrange(out, output_fold_eq, h=h, w=w)

        return out


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
            self,
            *,
            dim,
            hidden_dim=None,
            mix='ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing', 'midgoing', 'reverse_midgoing'}, 'mix must be ingoing, outgoing or midgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
        elif mix == 'midgoing':
            self.mix_einsum_eq = '... i k d, ... k j d -> ... i j d'
        elif mix == 'reverse_midgoing':
            self.mix_einsum_eq = '... j k d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


# evoformer blocks

class OuterMean(nn.Module):
    def __init__(
            self,
            dim,
            pair_dim,
            hidden_dim=None,
            eps=1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        pair_dim = default(pair_dim, dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, pair_dim)

    def forward(self, x, mask=None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')

        if exists(mask):
            # masked mean, if there are padding in the rows of the MSA
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)

        return self.proj_out(outer)


class PairwiseAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            pair_dim,
            seq_len,
            heads,
            dim_head,
            dropout=0.,
            global_column_attn=False
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim, pair_dim)

        # self.triangle_attention_outgoing = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=True,
        #                                                   col_attn=False, accept_edges=True)
        # self.triangle_attention_ingoing = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=False,
        #                                                  col_attn=True, accept_edges=True,
        #                                                  global_query_attn=global_column_attn)
        # self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=dim, mix='outgoing')
        # self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=dim, mix='ingoing')
        self.triangle_multiply_midgoing = TriangleMultiplicativeModule(dim=pair_dim, mix='midgoing')
        self.triangle_multiply_reverse_midgoing = TriangleMultiplicativeModule(dim=pair_dim, mix='reverse_midgoing')

    def forward(
            self,
            x,
            mask=None,
            msa_repr=None,
            msa_mask=None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask=msa_mask)

        x = self.triangle_multiply_midgoing(x, mask=mask) + x
        x = self.triangle_multiply_reverse_midgoing(x, mask=mask) + x
        # x = self.triangle_multiply_outgoing(x, mask=mask) + x
        # x = self.triangle_multiply_ingoing(x, mask=mask) + x
        # x = self.triangle_attention_outgoing(x, edges=x, mask=mask) + x
        # x = self.triangle_attention_ingoing(x, edges=x, mask=mask) + x
        return x


class MsaAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            pair_dim,
            seq_len,
            heads,
            dim_head,
            dropout=0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False,
                                       accept_edges=True, edge_dim=pair_dim)

    def forward(
            self,
            x,
            mask=None,
            pairwise_repr=None
    ):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr) + x
        return x


# main evoformer class

class EvoformerBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            pair_dim,
            seq_len,
            heads,
            dim_head,
            attn_dropout,
            ff_dropout,
            global_column_attn=False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim=dim, pair_dim=pair_dim, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                                   global_column_attn=global_column_attn),
            FeedForward(dim=pair_dim, dropout=ff_dropout),
            MsaAttentionBlock(dim=dim, pair_dim=pair_dim, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout),
            FeedForward(dim=dim, dropout=ff_dropout),
        ])

    def forward(self, inputs):
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # pairwise attention and transition
        # no token sequence experiment
        # x = attn(x, mask=mask, msa_repr=None, msa_mask=None)
        x = attn(x, mask=mask, msa_repr=m, msa_mask=msa_mask)
        x = ff(x) + x

        # msa attention and transition
        # no relation experiment
        # m = msa_attn(m, mask=msa_mask, pairwise_repr=None)
        m = msa_attn(m, mask=msa_mask, pairwise_repr=x)
        m = msa_ff(m) + m

        return x, m, mask, msa_mask


class Evoformer(nn.Module):
    def __init__(
            self,
            *,
            depth,
            **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
            self,
            x,
            m,
            mask=None,
            msa_mask=None
    ):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m