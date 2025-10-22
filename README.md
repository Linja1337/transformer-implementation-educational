# Transformer Architecture Implementation from Scratch

Educational deep dive into transformer internals through full PyTorch implementation of decoder-only language model with custom BPE tokenization, multi-head attention, and causal language modeling.

---

## Project Motivation

This project was undertaken to develop a foundational understanding of transformer architectures by implementing core components from scratch, rather than relying solely on high-level libraries. While production NLP work leverages frameworks like Hugging Face Transformers (see my Gaming Console Sentiment Analysis project), understanding the underlying mechanics enables better model optimization, debugging, and architectural decision-making.

**Key Learning Goal:** Master transformer fundamentals to become a more effective practitioner when using modern LLM frameworks.

---

## Architecture Overview

### Custom Components Implemented

**1. Byte Pair Encoding (BPE) Tokenizer**
- Subword tokenization algorithm from scratch
- Merge operation with frequency-based vocabulary building
- Special token handling for chat applications (<user>, <assistant>, <system>)
- Vocabulary: 10,000 tokens with configurable size

**2. Transformer Decoder Architecture**
- Multi-head self-attention with causal masking
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Sinusoidal positional embeddings
- Local attention windows for efficiency

**3. Training Infrastructure**
- Custom PyTorch Dataset for causal language modeling
- Mixed precision training (AMP) support
- Gradient accumulation for memory efficiency
- Checkpoint management and model saving
- TensorBoard logging

**4. Text Generation**
- Greedy decoding
- Temperature-based sampling
- Top-k sampling
- Nucleus (top-p) sampling
- Beam search

---

## Technical Implementation

### Model Configuration

```python
class TinyStoriesConfig:
    vocab_size: 10000          # BPE vocabulary size
    hidden_size: 256           # Embedding dimension
    num_hidden_layers: 4       # Transformer blocks
    num_attention_heads: 8     # Multi-head attention
    intermediate_size: 1024    # FFN hidden dimension
    max_position_embeddings: 512
    window_size: 256           # Local attention window
    dropout: 0.1
```

**Total Parameters:** ~20M (deliberately small for educational purposes)

### Training Details

**Dataset:** TinyStories (Hugging Face)
- Simple stories designed for language modeling research
- Controlled vocabulary and structure
- Enables faster experimentation on architecture components

**Training Setup:**
- Optimizer: AdamW with weight decay
- Learning rate: 3e-4 with linear warmup
- Batch size: 32 (effective with gradient accumulation)
- Mixed precision training for GPU efficiency
- Early stopping based on validation loss

---

## Key Learning Outcomes

### 1. Multi-Head Attention Mechanics
- Implemented scaled dot-product attention from scratch
- Understanding of query, key, value transformations
- Causal masking for autoregressive generation
- Attention weight computation and interpretation

### 2. Position Encoding Strategies
- Absolute positional embeddings (learned)
- Sinusoidal position encoding exploration
- Impact on sequence length generalization

### 3. Training Dynamics
- Gradient flow through deep architectures
- Layer normalization placement effects
- Residual connections for stable training
- Hyperparameter sensitivity analysis

### 4. Generation Algorithms
- Greedy vs sampling trade-offs
- Temperature effects on diversity
- Top-k and nucleus sampling comparison
- Beam search implementation challenges

### 5. Tokenization Impact
- BPE merge algorithm mechanics
- Vocabulary size vs model performance
- Handling of out-of-vocabulary words
- Special tokens for task-specific fine-tuning

---

## Project Components

### Core Implementation Files

**bpe_tokenizer.py**
- Custom BPE tokenizer with merge operations
- Encoding/decoding with special token support
- Serialization for model deployment

**transformer_model.py**
- Complete transformer decoder implementation
- Self-attention layers with causal masking
- Feed-forward networks with GELU activation
- Layer normalization and residual connections

**train_tinystories_model.py**
- Training pipeline for base language model
- Data loading from Hugging Face datasets
- Loss computation and optimization
- Checkpoint management

**train_tinystories_chat_model.py**
- Instruction tuning for conversational format
- Handling of multi-turn dialogue structure
- User/assistant token integration

**generate_tinystories_text.py**
- Text generation with multiple decoding strategies
- Command-line interface for inference

**chat_with_tinystories_model.py**
- Interactive chat interface
- Conversation history management
- Real-time response generation

### Analysis Notebooks

**data_exploration.ipynb**
- Dataset statistics and vocabulary analysis
- Token distribution visualization
- Sequence length characteristics

**bpe_exploration.ipynb**
- BPE merge operation visualization
- Vocabulary construction process
- Tokenization examples and edge cases

**pretrained_model_analysis.ipynb**
- Model checkpoint inspection
- Attention weight visualization
- Generated text quality assessment

**chat_model_analysis.ipynb**
- Instruction tuning evaluation
- Multi-turn conversation examples
- Response quality analysis

---

## Skills Demonstrated

### Deep Learning Engineering
- PyTorch model implementation from architectural papers
- Custom training loops with gradient accumulation
- Mixed precision training for efficiency
- Device management (CPU/CUDA/MPS)

### NLP Fundamentals
- Tokenization algorithm implementation
- Sequence modeling with attention mechanisms
- Causal language modeling objective
- Text generation algorithms

### Software Engineering
- Modular, reusable code architecture
- Command-line interfaces with argparse
- Model serialization and loading
- Comprehensive documentation

### Research Methodology
- Ablation studies on architectural components
- Hyperparameter experimentation
- Qualitative and quantitative evaluation
- Notebook-based analysis and visualization

---

## Relationship to Production Work

**This implementation complements applied LLM work:**

**When to build from scratch (this project):**
- Understanding model internals for debugging
- Experimenting with novel architectural components
- Educational purposes and knowledge building
- Research on attention mechanisms

**When to use frameworks (Gaming Console project):**
- Production applications with real business value
- Fine-tuning pre-trained models like BERT, GPT
- Leveraging state-of-the-art architectures
- Rapid prototyping and deployment

**Key Insight:** Understanding foundational mechanics (this project) makes framework usage (Hugging Face) more effective through informed hyperparameter tuning, architecture selection, and debugging capabilities.

---

## Running the Code

### Installation

```bash
pip install torch datasets tqdm numpy tensorboard
```

### Training BPE Tokenizer

```bash
python train_bpe_tokenizer_hf.py --sample 10000
```

### Training Base Language Model

```bash
python train_tinystories_model.py \
    --dataset roneneldan/TinyStories \
    --output_dir tinystories_model \
    --num_epochs 10 \
    --batch_size 32
```

### Text Generation

```bash
python generate_tinystories_text.py \
    --model_path tinystories_model/best_model.pth \
    --prompt "Once upon a time" \
    --temperature 0.8
```

### Interactive Chat

```bash
python chat_with_tinystories_model.py \
    --model_path tinystories_chat_model/final_model.pth \
    --temperature 0.7
```

---

## Technical Stack

**Framework & Libraries:**
- PyTorch (model implementation, training)
- Hugging Face Datasets (data loading)
- NumPy (numerical operations)
- TensorBoard (training visualization)

**Key Techniques:**
- Multi-head self-attention with causal masking
- Byte Pair Encoding tokenization
- AdamW optimization with weight decay
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling

---

## Limitations and Future Work

### Current Limitations
- Small model size (20M parameters) for educational purposes
- TinyStories dataset is simplified for faster experimentation
- Limited to decoder-only architecture (no encoder-decoder)
- Basic generation algorithms (no advanced techniques like MCTS)

### Potential Extensions
1. Scale to larger models (100M+ parameters)
2. Implement encoder-decoder architecture for sequence-to-sequence tasks
3. Add advanced generation techniques (beam search with constraints)
4. Explore alternative position encoding schemes (RoPE, ALiBi)
5. Implement efficient attention variants (Flash Attention, sparse attention)
6. Multi-GPU distributed training
7. Quantization and model compression techniques

---

## Key Takeaways

**What This Project Demonstrates:**
- Deep understanding of transformer architecture internals
- Ability to implement complex neural architectures from papers
- Strong PyTorch and deep learning engineering skills
- Foundation for effectively using modern LLM frameworks

**What This Project Is NOT:**
- Production-ready language model
- State-of-the-art performance (intentionally simplified)
- Replacement for Hugging Face Transformers in applied work
- Scalable to enterprise applications

**Educational Value:**
This implementation serves as a foundation for understanding how modern LLMs work internally, enabling more informed decisions when using production frameworks like Hugging Face Transformers for real-world applications (sentiment analysis, classification, document understanding).

---

## References

**Foundational Papers:**
- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
- Radford et al. (2018). "Improving Language Understanding by Generative Pre-Training." (GPT)
- Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." (BPE)

**Dataset:**
- Eldan & Li (2023). "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"

---

**Note:** This project was created for educational purposes to master transformer fundamentals. For production NLP applications, see my Gaming Console Sentiment Analysis project which demonstrates applied LLM fine-tuning with Hugging Face Transformers on real-world business problems.
