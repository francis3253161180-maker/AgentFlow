"""Torch-only subset of flash_attn.bert_padding used by verl's FSDP path.

The smoke run uses PyTorch SDPA rather than FlashAttention CUDA kernels.  verl
still imports these padding utilities, so keep their public behavior available
without compiling an unrelated multi-architecture CUDA extension.
"""

from einops import rearrange
import torch
import torch.nn.functional as F


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices]


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    unpadded = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
    return unpadded, indices, cu_seqlens, max_seqlen


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])
