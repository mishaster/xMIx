# xmix synthesized from vllm/activations_extractor/applications/xmix_examples/app1.xmix
# assumes in scope: model, torch

# === SETUP (once, e.g. in __init__) ===

# --- user preamble (verbatim) ---
self.vec_indices = torch.full((2048,),1, dtype=torch.int32, device=self.device)
self.arbitrary_vector = torch.full((self.hidden_size,), 0.1,device=self.device, dtype=self.model_config.dtype)
self.steer_refusal_vec = SteeringVectorDotSubtractNormalized(hidden_size = self.model_config.hidden_size, max_tokens = 2048,
    steering_vector = self.arbitrary_vector, vec_indices = self.vec_indices, dtype = self.model_config.dtype, device = self.device)

# vllm/activations_extractor/applications/xmix_examples/app1.xmix:5  m.write(self.steer_refusal_vec.run).layer("all").submodule("mlp.post")

# === INSTALL (after load_model) ===

# steer SteeringVectorDotSubtractNormalized on ALL layers at 'mlp.post' (flag 'w')
for layer in model.model.layers:
    layer.set_post_mlp_hook(self.steer_refusal_vec.run, "w")
