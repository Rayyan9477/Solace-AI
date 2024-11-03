import torch
from safetensors import safe_open
import argparse

def convert_safetensors_to_bin(input_path, output_path):
    """
    Convert a model from .safetensors format to .bin format.

    Parameters:
    input_path (str): Path to the input .safetensors file.
    output_path (str): Path where the output .bin file will be saved.
    """
    tensors = {}

    # Load the tensors from .safetensors file
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Save the tensors in .bin format
    torch.save(tensors, output_path)
    print(f"Model converted successfully and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .safetensors model to .bin format")
    parser.add_argument('input_path', type=str, help='Path to the input .safetensors file')
    parser.add_argument('output_path', type=str, help='Path to save the output .bin file')
    args = parser.parse_args()
    convert_safetensors_to_bin(args.input_path, args.output_path)