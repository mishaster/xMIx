# xmix synthesized from vllm/activations_extractor/applications/xmix_examples/steer.xmix
# assumes in scope: LinearProbe, SteeringVectorAdder, _cond_bias, _max_tokens, _steered_tokens_num, arg1, biggerThan, cond_arg1, cond_arg2, model, torch

# === SETUP (once, e.g. in __init__) ===

# vllm/activations_extractor/applications/xmix_examples/steer.xmix:1  m.write(SteeringVector(arg1).run).layers([23, 24, 25]).submodule("attention.post")
#               .cond(LinearProbe.biggerThan(cond_arg1, cond_arg2).layer([18]).submodule("mlp.post"))
_r0 = arg1.unsqueeze(0) if arg1.ndim == 1 else arg1
_max_vecs0, _hidden0 = _r0.shape
_scales0 = torch.ones((_max_tokens, _max_vecs0), dtype=_r0.dtype, device=_r0.device)
_vec_indices0 = torch.arange(_max_vecs0, dtype=torch.int32, device=_r0.device).repeat(_max_tokens, 1)
_n_vecs_per_token0 = torch.full((_max_tokens,), _max_vecs0, dtype=torch.int32, device=_r0.device)
sv_0 = SteeringVectorAdder(_r0, _scales0, _vec_indices0, _n_vecs_per_token0)
lp_0 = LinearProbe(cond_arg1, _cond_bias, _max_tokens)
cond_0 = biggerThan(lp_0, cond_arg2, sv_0.input_map)   # user-defined probs->mask bridge

# === INSTALL (after load_model) ===

# steer SteeringVector on layers [23, 24, 25] at 'attention.post' (flag 'w')
for layer in model.model.layers:
    idx = layer.self_attn.layer_idx
    if idx in (23, 24, 25):
        layer.set_post_attn_pre_norm_hook(lambda _a0: sv_0.run(_a0, sv_0.r, _steered_tokens_num), "w")
    if idx in (18,):
        layer.set_post_mlp_hook(cond_0, "r")
