#!/usr/bin/env python3
"""
Model Quantization for Edge Deployment
Reduces model size and speeds up inference on Raspberry Pi
"""

import argparse
import torch
from pathlib import Path
import logging

from custom_asr import SushiASRModel, create_character_vocab


def quantize_model_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply dynamic quantization to model
    
    Quantizes LSTM and Linear layers to int8
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.LSTM, torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='Quantize SushiVoice model for edge deployment')
    parser.add_argument('--input', type=str, required=True,
                        help='Input model checkpoint path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output quantized model path')
    parser.add_argument('--test', action='store_true',
                        help='Test quantized model')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_quantized.pth")
    
    logging.info(f"Loading model from {args.input}...")
    checkpoint = torch.load(args.input, map_location='cpu')
    
    # Get vocabulary
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
        char_to_idx = checkpoint['char_to_idx']
    else:
        vocab, char_to_idx = create_character_vocab()
    
    # Create model
    model = SushiASRModel(
        vocab_size=len(vocab),
        n_cnn_layers=4,
        n_rnn_layers=3,
        rnn_hidden_size=384,
        use_spec_features=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Original model size: {model.get_param_count():,} parameters")
    
    # Quantize
    logging.info("Applying dynamic quantization...")
    quantized_model = quantize_model_dynamic(model)
    
    # Save quantized model
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'vocab': vocab,
        'char_to_idx': char_to_idx,
        'quantized': True
    }, args.output)
    
    # Get file sizes
    original_size = Path(args.input).stat().st_size / (1024 * 1024)
    quantized_size = Path(args.output).stat().st_size / (1024 * 1024)
    
    logging.info(f"Quantized model saved to {args.output}")
    logging.info(f"Original size: {original_size:.2f} MB")
    logging.info(f"Quantized size: {quantized_size:.2f} MB")
    logging.info(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Test inference
    if args.test:
        logging.info("\nTesting inference speed...")
        
        # Test input
        test_input = torch.randn(1, 1, 48000)  # 3 seconds
        
        # Original model
        import time
        model.eval()
        with torch.no_grad():
            start = time.time()
            _ = model(test_input)
            original_time = time.time() - start
        
        # Quantized model
        quantized_model.eval()
        with torch.no_grad():
            start = time.time()
            _ = quantized_model(test_input)
            quantized_time = time.time() - start
        
        logging.info(f"Original inference time: {original_time*1000:.1f}ms")
        logging.info(f"Quantized inference time: {quantized_time*1000:.1f}ms")
        logging.info(f"Speedup: {original_time/quantized_time:.2f}x")
    
    print("\n✅ Quantization complete!")


if __name__ == '__main__':
    main()
