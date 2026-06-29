import sys
from transformers import AutoTokenizer

def main():
    # 1. Change this to the model you are currently reviewing in vLLM
    #deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    #model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
    
    print(f"--- Loading Tokenizer for {model_id} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Vocabulary Size: {len(tokenizer)}")
    except Exception as e:
        print(f"Error: Could not load model. Make sure you have access and 'transformers' installed.\n{e}")
        return

    print("\nEnter Token IDs (space-separated) or 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nIDs > ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Convert input string to a list of integers
            ids = [int(i) for i in user_input.replace(',', ' ').split()]
            
            # Translate to string
            # skip_special_tokens=False is useful to see things like <|endoftext|>
            decoded_text = tokenizer.decode(ids, skip_special_tokens=False)
            
            # Also show individual token fragments for debugging
            fragments = [tokenizer.decode([i]) for i in ids]
            
            print(f"Full Text: '{decoded_text}'")
            print(f"Fragments: {fragments}")
            
        except ValueError:
            print("Invalid input. Please enter numbers only.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
