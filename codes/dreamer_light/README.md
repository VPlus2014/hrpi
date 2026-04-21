# Dreamer Light

A lightweight version of DreamerV3 optimized for vectorized observations, stripped of image processing components.

## Features

- Simplified RSSM (Recurrent State-Space Model)
- MLP-based encoder and decoder for vector observations
- Reward and continuation predictors
- Policy and value networks
- JAX-based implementation

## Usage

1. Install dependencies:

   ```bash
   pip install embodied jax ninjax elements
   ```

2. Run training:

   ```bash
   python -m dreamer_light.train
   ```

## Configuration

Edit `configs.yaml` to adjust model sizes and training parameters.

## Differences from DreamerV3

- Removed image-specific CNN layers
- Reduced model sizes for lightweight deployment
- Focused on vectorized observation spaces
- Simplified configuration
