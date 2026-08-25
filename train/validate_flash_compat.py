import torch
from flash_attn.bert_padding import pad_input, unpad_input

x = torch.randn(2, 3, 4)
mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
unpadded, indices, cu_seqlens, max_seqlen = unpad_input(x, mask)
restored = pad_input(unpadded, indices, 2, 3)
assert unpadded.shape == (3, 4)
assert indices.tolist() == [0, 1, 3]
assert cu_seqlens.tolist() == [0, 2, 3]
assert max_seqlen == 2
assert torch.equal(restored[mask.bool()], x[mask.bool()])
print("flash_attn_compat_ok")
