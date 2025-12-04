import time
from mlx_lm import load, generate

# Configuration
model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
prompt = "Write a single sentence interesting fact about space."
max_tokens = 100

def main():
    print(f"Loading model: {model_path}...")
    # Load model and tokenizer (this happens only once)
    model, tokenizer = load(model_path)
    print("Model loaded. Starting loop...\n")

    try:
        while True:
            print("-" * 40)
            print(f"Prompt: {prompt}")
            
            # Start timer
            start_time = time.time()
            
            # Generate response
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=max_tokens, 
                verbose=False # We print manually to calculate precise TPS
            )
            
            # End timer
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate metrics
            token_count = len(tokenizer.encode(response))
            tps = token_count / duration
            
            print(f"\nResponse: {response.strip()}")
            print(f"\nStats: {token_count} tokens in {duration:.2f}s")
            print(f"Speed: \033[1;32m{tps:.2f} tokens/sec\033[0m")
            
            # Small pause to make it readable
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nLoop stopped by user.")

if __name__ == "__main__":
    main()