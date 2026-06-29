import torch

# Replace with your actual path
#vector_path = "/home/michael/refusal_direction/pipeline/runs/meta-llama-3-8b-instruct/direction.pt"
vector_path = "/home/michael/vllm/vllm/activations_extractor/applications/reflection_avg_vector.pt"
refusal_vector = torch.load(vector_path)

print(refusal_vector.shape) # Should match the model's hidden dimension
print(refusal_vector)
