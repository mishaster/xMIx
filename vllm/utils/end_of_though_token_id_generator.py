from transformers import AutoTokenizer
import json

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Find every token ID containing the double newline
newline_ids = [
    i for i in range(len(tokenizer)) 
    if "\n\n" in tokenizer.decode([i], clean_up_tokenization_spaces=False)
]

print(f"Detected {len(newline_ids)} tokens containing '\\n\\n'.")
print("Copy this list into your gpu_model_runner.py:")
print(newline_ids)
