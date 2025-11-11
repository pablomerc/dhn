# Extractor.py Explanation

## Purpose

`extractor.py` is responsible for **optimizing latent code embeddings (the codebook) for specific data sequences** while keeping the rest of the pre-trained Hamiltonian neural network frozen.

## How It Works

### Key Functionality

1. **Loads Pre-trained Model**: The extractor loads a checkpoint from a previously trained model (created by `trainer.py`)

2. **Freezes Network Parameters**: All network parameters are frozen except for the codebook embeddings
   - The `trainable_prefix_list = ['codebook']` specifies that only codebook parameters should be trainable
   - All other parameters are loaded and then set to `requires_grad = False`

3. **Optimizes Latent Codes**: The extractor fine-tunes the codebook embeddings to better represent the specific data sequences in the dataset
   - Uses `extract_get_losses()` method which wraps the standard `get_losses()` method
   - Uses `extract_inference()` method which wraps the standard `inference()` method

4. **Saves Optimized Codes**: The optimized codebook is saved to a separate directory (`extract_dir`) for later use

## Relationship to Other Components

### Workflow Pipeline

```
1. trainer.py → Train the full Hamiltonian neural network from scratch
                (all parameters are trainable)
                ↓
2. extractor.py → Optimize latent codes for specific sequences
                  (only codebook is trainable, rest is frozen)
                  ↓
3. generator.py → Generate new pendulum sequences using the trained model
                  (with optimized codes)
```

### Comparison Table

| Component | Purpose | What's Trainable | Input | Output |
|-----------|---------|------------------|-------|--------|
| **trainer.py** | Train full model | All parameters | Training data | Trained model checkpoint |
| **extractor.py** | Optimize latent codes | Only codebook | Pre-trained model + data | Optimized codebook |
| **generator.py** | Generate sequences | Nothing (inference only) | Trained model + codes | Generated sequences |

## Technical Details

### Codebook Function

The codebook is an embedding layer that maps sequence indices to latent codes:
- `get_latent_code(data)` → `self.codebook(data['idx'])`
- Each sequence in the dataset has an associated index that maps to a latent code vector
- These latent codes are used as conditioning information for the Hamiltonian dynamics

### Key Methods

- `load_pretrained_checkpoint()`: Loads the trained model but filters out codebook parameters, then freezes all loaded parameters
- `train_step()`: Optimizes the codebook using the test loader data
- `extract_get_losses()`: Computes loss for codebook optimization
- `extract_inference()`: Runs inference to evaluate the optimized codes

## Why This Approach?

This two-stage training approach (full training → codebook optimization) allows:

1. **Better Representation**: The codebook can be fine-tuned to better capture sequence-specific characteristics
2. **Preserved Dynamics**: The learned Hamiltonian dynamics remain unchanged, ensuring physical consistency
3. **Flexibility**: Different codebooks can be optimized for different datasets or use cases while reusing the same trained dynamics model
