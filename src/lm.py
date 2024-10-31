import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
from sample import SynthIdSampler

class LanguageModel:
    def __init__(self, credentials_path='credentials.json'):
        """Initialize the Language Model with credentials from a JSON file."""
        # Load credentials
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
        
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        
        # Extract API key for Hugging Face
        self.api_key = credentials.get('api_key')
        if not self.api_key:
            raise ValueError("API key not found in credentials.")
        
        # Set the Hugging Face API token as an environment variable
        os.environ['HUGGINGFACE_TOKEN'] = self.api_key

        # Initialize model name (to be provided via command-line argument)
        self.model_name = None  # Will be set externally

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None

    def load_model(self, model_name):
        """Load the tokenizer and model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.api_key)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=self.api_key)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def generate(self, input_text, max_length=50):
        """Generate text from the model based on input text."""
        if not self.tokenizer or not self.model:
            raise RuntimeError("Model not loaded. Call load_model() before generate().")
        
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        breakpoint()
        logits = self.model(inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(logits[0], skip_special_tokens=True)

    def serve_requests(self, **args):
        while True:
            inp = input("Enter input text, or 'exit' to quit: ")
            if inp == 'exit':
                break
            print(self.generate(inp, **args))


class WatermarkedLM(LanguageModel):
    def __init__(self, model_name, hf_token, watermarker):
        super().__init__(hf_token)
        super().load_model(model_name)
        self.watermarker = watermarker

    def generate(self, input_text, max_length=None, do_watermark=False):
        if do_watermark:
            raise NotImplementedError("Watermarking not implemented yet")
        else:
            return super().generate(input_text, max_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a Hugging Face model.")
    parser.add_argument("model_name", type=str, help="The name of the Hugging Face model to use.")
    parser.add_argument("--credentials_path", type=str, default="credentials.json", help="Path to the credentials JSON file.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")
    parser.add_argument("--watermarker", type=str, default=None, help="The name of the watermarker to use.")

    args = parser.parse_args()

    if args.watermarker == 'SynthId':
        watermarker = SynthIdSampler()
    else:
        raise ValueError(f"Watermarker {args.watermarker} not supported")
    
    lm = WatermarkedLM(args.model_name, args.credentials_path, watermarker)
    lm.serve_requests()
