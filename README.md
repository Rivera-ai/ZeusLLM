# ZeusLLM ⚡️

Implementation of a Large Language Model based on the Llama 3 architecture, focusing on efficiency and scalability.

## 🌟 Features

- **Llama 3 Architecture**: Implements key components of the Llama 3 architecture including RoPE (Rotary Position Embedding), RMSNorm, and Multi-Query Attention
- **Mixed Precision Training**: Utilizes PyTorch's automatic mixed precision for efficient training
- **Flexible Data Processing**: Supports large-scale dataset preparation with efficient memory mapping
- **TensorBoard Integration**: Built-in training monitoring and visualization
- **Configurable Architecture**: Easily adjustable model parameters including dimensions, layers, and attention heads
- **Inference Pipeline**: Ready-to-use inference system with temperature and top-k/p sampling controls

## 🚀 Installation

```bash
git clone https://github.com/[your-username]/ZeusLLM.git
cd ZeusLLM
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
transformers
numpy
tqdm
tensorboard
datasets
```

## 📚 Project Structure

```
ZeusLLM/
├── llm.py             # Core model architecture implementation
├── preparedata.py     # Data preprocessing and tokenization
├── train.py           # Training loop and utilities
├── inference.py       # Inference and text generation
├── data/              # Directory for processed datasets
└── outputs/           # Model checkpoints and TensorBoard logs
```

## 💻 Usage

### 1. Data Preparation

Process and tokenize your training data:

```bash
python preparedata.py --model-name "meta-llama/Llama-3.1-8B" \
                     --max-seq-length 2048 \
                     --num-proc 8 \
                     --output-dir "data" \
                     --val-size 0.0005
```

### 2. Training

Train the model:

```bash
python train.py
```

Key training features:
- Automatic mixed precision training
- TensorBoard logging
- Checkpoint saving
- Validation monitoring

### 3. Inference

Generate text using a trained model:

```python
from inference import LLMInference

generator = LLMInference("outputs/best_model.pt")
response = generator.generate(
    prompt="En un futuro lejano,",
    max_new_tokens=20,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1
)
print(response)
```

## 🔧 Model Architecture

The model implements the following key components:

- **Attention Mechanism**:
  - Multi-query attention for efficient inference
  - Configurable number of heads and KV heads
  - Rotary Position Embeddings (RoPE)

- **Feed Forward Network**:
  - SwiGLU activation
  - Configurable hidden dimensions
  - Dropout for regularization

- **Normalization**:
  - RMSNorm for stable training
  - Configurable epsilon parameter

### Default Configuration

```python
ModelArgs(
    dim=768,              # Model dimension
    n_layers=12,          # Number of transformer layers
    n_heads=12,           # Number of attention heads
    vocab_size=32000,     # Vocabulary size
    max_seq_len=2048,     # Maximum sequence length
    dropout=0.1           # Dropout rate
)
```

## 📊 Training Details

The training pipeline includes:

- Gradient scaling for mixed precision
- Configurable batch sizes and learning rates
- TensorBoard metrics tracking:
  - Training/validation loss
  - Learning rate schedules
  - Model gradients
- Automatic model checkpointing

## 📄 License

This project is licensed under the RiveraAICloseLicense - see the [LICENSE](LICENSE.txt) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- The Llama team at Meta AI for the original architecture
- The PyTorch team for their excellent framework
- The Hugging Face team for their transformers library
