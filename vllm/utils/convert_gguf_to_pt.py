import torch
from gguf import GGUFReader

# 1. Point the reader to your GGUF file
#file_path = "~/EasySteer/replications/seal/reflection_avg_vector.gguf"
#file_path = "/home/michael/EasySteer/replications/seal/reflection_avg_vector.gguf"
file_path = "/home/michael/vllm/vllm/activations_extractor/applications/execution_avg_vector.gguf"
reader = GGUFReader(file_path)

# 2. Loop through the tensors in the file (there is likely only one for an avg_vector)
for tensor in reader.tensors:
    print(f"Found vector: {tensor.name} with shape {tensor.shape}")
    
    # 3. The magic step: gguf loads the data as a standard NumPy array. 
    # We simply wrap it in a PyTorch tensor.
    pytorch_vector = torch.tensor(tensor.data)
    
    # 4. (Optional) Save it as a standard .pt file so you can use torch.load() later
    output_filename = "transition_avg_vector.pt"
    torch.save(pytorch_vector, output_filename)
    
    print(f"Success! Saved as {output_filename}")
