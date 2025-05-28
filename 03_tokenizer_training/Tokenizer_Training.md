# Tokenizer Training 

Professional tokenizer training system for building custom tokenizers optimized for your domain and data.

## 📁 Contents

### 📚 Documentation
- **[TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md)** - Comprehensive guide covering all tokenizer types, training methods, and best practices
- **[README.md](README.md)** - This overview file

### 🔧 Core Scripts
- **[train_tokenizer.py](train_tokenizer.py)** - Production tokenizer training (50K vocab, all data)
- **[test_tokenizer_small.py](test_tokenizer_small.py)** - Quick test training (8K vocab, subset data)
- **[setup_and_test.py](setup_and_test.py)** - Environment validation and setup testing

### 🔍 Analysis Tools
- **[tokenizer_comparison.py](tokenizer_comparison.py)** - Compare different tokenizer types with performance analysis

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Install dependencies
pip install -r ../requirements.txt

# Validate setup
python setup_and_test.py
```

### 2. **Quick Test Training**
```bash
# Train small tokenizer for testing
python test_tokenizer_small.py
```

### 3. **Production Training**
```bash
# Train full production tokenizer
python train_tokenizer.py
```

### 4. **Compare Tokenizers**
```bash
# Compare different tokenizer types
python tokenizer_comparison.py
```

## 📊 What You'll Get

### Tokenizer Types Supported
- **BPE (Byte-Pair Encoding)** - Used by GPT-2, GPT-3, RoBERTa
- **WordPiece** - Used by BERT, DistilBERT, Electra  
- **Unigram** - Used by XLNet, ALBERT, T5, mBART
- **Custom Domain-Specific** - Optimized for your specific use case

### Training Configurations
- **Testing**: 8K vocabulary, 2 files, ~5 minutes
- **Production**: 50K vocabulary, all data, ~30-60 minutes
- **Custom**: Configurable vocabulary size and domain optimization

### Output Formats
- **Hugging Face Transformers** compatible
- **JSON configuration** files
- **Vocabulary and merge** files
- **Performance metrics** and analysis

## 🎯 Use Cases

### Domain-Specific Applications
- **Medical/Scientific** - Preserve technical terminology
- **Legal** - Handle complex legal language
- **Code** - Optimize for programming languages
- **Multilingual** - Support multiple languages efficiently

### Performance Optimization
- **Memory Efficiency** - Reduce model parameter count
- **Speed Optimization** - Faster tokenization and inference
- **Compression** - Better text-to-token ratios

## 📈 Performance Expectations

### Training Time
- **Small test**: 2-5 minutes (8K vocab)
- **Medium**: 15-30 minutes (25K vocab)
- **Production**: 30-60 minutes (50K vocab)
- **Large multilingual**: 2-4 hours (100K vocab)

### Data Requirements
- **Minimum**: 100MB cleaned text
- **Recommended**: 1GB+ for good quality
- **Optimal**: 10GB+ for production quality
- **Multilingual**: 50GB+ for comprehensive coverage

### Quality Metrics
- **Compression ratio**: 3.5-4.5 chars/token (English)
- **Coverage**: >98% known tokens on domain text
- **Speed**: <1ms tokenization time for typical sentences

## 🔗 Integration

### With Data Pipeline
1. **Data Collection** → `01_data_collection/`
2. **Data Processing** → `02_data_processing/` 
3. **Tokenizer Training** → `03_tokenizer_training/` (You are here)
4. **Data Tokenization** → `04_data_tokenization/`
5. **Model Training** → `05_model_training/`

### With ML Frameworks
```python
# Load trained tokenizer
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("./my_tokenizer")

# Use in model training
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))
```

## 📋 File Structure After Training

```
03_tokenizer_training/
├── models/
│   └── my_tokenizer/
│       ├── vocab.json              # Vocabulary mapping
│       ├── merges.txt             # BPE merge rules
│       ├── tokenizer.json         # Complete config
│       ├── config.json            # Transformers config
│       └── special_tokens_map.json # Special tokens
├── test_tokenizer_output/         # Test outputs
├── logs/                          # Training logs
└── data/
    └── sample/                    # Sample training data
```

## 🛠️ Troubleshooting

### Common Issues
- **Out of memory**: Reduce batch size or use chunked processing
- **Poor quality**: Increase training data or vocabulary size
- **Slow training**: Use smaller vocabulary for testing
- **Integration errors**: Check Transformers version compatibility

### Getting Help
1. Check [TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md) for detailed solutions
2. Run `python setup_and_test.py` to validate environment
3. Start with small test before scaling up
4. Verify data format (JSONL with "text" field)

## 🎉 Success Criteria

You'll know your tokenizer is ready when:
- ✅ Training completes without errors
- ✅ Test tokenization produces reasonable tokens
- ✅ Compression ratio is 3.5+ chars/token
- ✅ Unknown token rate is <2% on your domain
- ✅ Integration with Transformers works smoothly

## 📖 Learn More

- **[TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md)** - Complete technical documentation
- **[Hugging Face Tokenizers](https://huggingface.co/learn/llm-course/en/chapter6/8)** - Official documentation
- **[tokenizer_comparison.py](tokenizer_comparison.py)** - Interactive examples and comparisons

---

**Ready to build better tokenizers?** Start with `python setup_and_test.py` to validate your environment! 