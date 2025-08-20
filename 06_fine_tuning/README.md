# 🎯 Fine-Tuning ReFocused-AI Model

This module provides comprehensive fine-tuning capabilities for the ReFocused-AI base model, supporting various task-specific adaptations including chat, code generation, instruction following, and domain-specific applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Tasks](#supported-tasks)
- [Quick Start](#quick-start)
- [Fine-Tuning Methods](#fine-tuning-methods)
- [Configuration](#configuration)
- [Dataset Preparation](#dataset-preparation)
- [Advanced Usage](#advanced-usage)
- [Monitoring & Evaluation](#monitoring--evaluation)
- [Best Practices](#best-practices)

## 🌟 Overview

The fine-tuning module allows you to adapt your pre-trained ReFocused-AI model for specific tasks while maintaining the general knowledge learned during pre-training. It supports:

- **Multiple fine-tuning strategies**: Full fine-tuning, LoRA, layer freezing
- **Task-specific optimizations**: Tailored configurations for different use cases
- **Efficient training**: Memory-optimized techniques for resource-constrained environments
- **Comprehensive evaluation**: Task-specific metrics and monitoring

## 🎯 Supported Tasks

### 1. **Chat Fine-tuning** (`--task chat`)
- Optimizes the model for conversational interactions
- Supports multi-turn dialogue formatting
- Includes response diversity metrics

### 2. **Code Fine-tuning** (`--task code`)
- Adapts the model for code generation and completion
- Supports multiple programming languages
- Includes syntax consistency metrics

### 3. **Instruction Fine-tuning** (`--task instruct`)
- Trains the model to follow specific instructions
- Supports various instruction formats
- Includes instruction adherence metrics

### 4. **Domain Fine-tuning** (`--task domain`)
- Specializes the model for specific domains (medical, legal, etc.)
- Extended training for thorough adaptation
- Domain-specific vocabulary enhancement

### 5. **Custom Fine-tuning** (`--task custom`)
- Flexible configuration for unique use cases
- Customizable data formatting
- General-purpose metrics

## 🚀 Quick Start

### Basic Fine-tuning

```bash
# Fine-tune for chat
python fine_tune.py \
    --task chat \
    --base-model ./05_model_training/checkpoints/final_model \
    --dataset ./datasets/chat_data.jsonl \
    --output-dir ./fine_tuned_models

# Fine-tune with LoRA (parameter-efficient)
python fine_tune.py \
    --task instruct \
    --base-model ./05_model_training/checkpoints/final_model \
    --dataset alpaca-instruct \
    --lora \
    --lora-rank 16 \
    --output-dir ./fine_tuned_models
```

### Using HuggingFace Datasets

```bash
# Fine-tune on HuggingFace dataset
python fine_tune.py \
    --task code \
    --base-model ./05_model_training/checkpoints/final_model \
    --dataset "codeparrot/github-code" \
    --output-dir ./fine_tuned_models
```

## 🔧 Fine-Tuning Methods

### 1. **Full Fine-tuning**
- Updates all model parameters
- Best for significant task shifts
- Requires more compute and memory

```bash
python fine_tune.py \
    --task domain \
    --base-model ./base_model \
    --dataset ./domain_data.jsonl
```

### 2. **LoRA (Low-Rank Adaptation)**
- Updates only a small number of parameters
- Memory efficient
- Maintains base model performance

```bash
python fine_tune.py \
    --task chat \
    --base-model ./base_model \
    --dataset ./chat_data.jsonl \
    --lora \
    --lora-rank 8
```

### 3. **Layer Freezing**
- Freezes bottom layers, fine-tunes top layers
- Balance between efficiency and adaptation
- Good for similar tasks

```bash
python fine_tune.py \
    --task instruct \
    --base-model ./base_model \
    --dataset ./instruct_data.jsonl \
    --freeze-ratio 0.7
```

## ⚙️ Configuration

### Preset Configurations

```bash
# Quick test run
python fine_tune.py \
    --task chat \
    --config quick_test \
    --base-model ./base_model \
    --dataset ./test_data.jsonl

# Low resource settings
python fine_tune.py \
    --task instruct \
    --config low_resource \
    --base-model ./base_model \
    --dataset ./data.jsonl

# High quality training
python fine_tune.py \
    --task domain \
    --config high_quality \
    --base-model ./base_model \
    --dataset ./domain_data.jsonl
```

### Custom Configuration

```python
# In configs/fine_tuning_config.py
custom_config = FineTuningConfig(
    learning_rate=1e-5,
    num_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    eval_steps=100,
    save_steps=500,
    max_length=2048,
)
```

## 📊 Dataset Preparation

### Format Examples

#### Chat Format (JSONL)
```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help you?"}]}
{"prompt": "What is machine learning?", "response": "Machine learning is..."}
```

#### Instruction Format (JSONL)
```json
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Summarize this text", "output": "This is a summary..."}
```

#### Code Format (JSONL)
```json
{"prompt": "Write a function to sort a list", "completion": "def sort_list(lst):\n    return sorted(lst)"}
{"code": "class DataProcessor:\n    def __init__(self):\n        pass"}
```

### Data Validation Script

```bash
# Validate your dataset
python scripts/validate_dataset.py \
    --dataset ./my_data.jsonl \
    --task chat
```

## 🚀 Advanced Usage

### Multi-GPU Training

```bash
# Distributed fine-tuning
accelerate launch fine_tune.py \
    --task chat \
    --base-model ./base_model \
    --dataset ./large_dataset.jsonl \
    --config production
```

### Gradient Checkpointing

```bash
# Enable for large models/long sequences
python fine_tune.py \
    --task code \
    --base-model ./base_model \
    --dataset ./code_data.jsonl \
    --gradient-checkpointing \
    --max-length 4096
```

### Mixed Precision Training

```bash
# Use bfloat16 for stable training
python fine_tune.py \
    --task domain \
    --base-model ./base_model \
    --dataset ./domain_data.jsonl \
    --mixed-precision bf16
```

### Resume Training

```bash
# Resume from checkpoint
python fine_tune.py \
    --task chat \
    --base-model ./base_model \
    --dataset ./data.jsonl \
    --resume checkpoint-step-1000
```

## 📈 Monitoring & Evaluation

### Training Metrics
- Loss curves
- Learning rate schedules
- Gradient norms
- Training/validation accuracy

### Task-Specific Metrics

#### Chat Metrics
- Response diversity
- Turn coherence
- Conversation flow

#### Code Metrics
- Syntax validity
- Code completion accuracy
- Pattern consistency

#### Instruction Metrics
- Instruction adherence
- Task completion rate
- Response quality

### Evaluation During Training

```bash
# Frequent evaluation
python fine_tune.py \
    --task instruct \
    --base-model ./base_model \
    --dataset ./data.jsonl \
    --eval-steps 50 \
    --save-steps 200
```

## 💡 Best Practices

### 1. **Data Quality**
- Ensure high-quality, task-relevant data
- Balance dataset size and diversity
- Validate data format before training

### 2. **Learning Rate Selection**
- Start with recommended rates per task
- Use learning rate finder for optimal values
- Consider warmup for stable training

### 3. **Batch Size Optimization**
- Larger batches for stable gradients
- Use gradient accumulation if memory-limited
- Monitor GPU utilization

### 4. **Regularization**
- Use dropout for overfitting prevention
- Apply weight decay appropriately
- Monitor train/validation gap

### 5. **Checkpoint Management**
- Save checkpoints frequently
- Keep best N checkpoints
- Enable checkpoint resumption

### 6. **Evaluation Strategy**
- Evaluate on held-out data
- Use task-specific metrics
- Monitor for overfitting

## 🛠️ Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--per-device-train-batch-size 1 \
--gradient-accumulation-steps 8

# Enable gradient checkpointing
--gradient-checkpointing

# Use LoRA
--lora --lora-rank 4
```

### Slow Training
```bash
# Enable mixed precision
--mixed-precision fp16

# Reduce sequence length
--max-length 512

# Use more workers
--num-workers 8
```

### Poor Performance
```bash
# Increase training time
--num-epochs 10

# Adjust learning rate
--learning-rate 1e-6

# Use larger LoRA rank
--lora-rank 32
```

## 📦 Output Structure

```
fine_tuned_models/
├── chat_chat_data/
│   ├── final_model/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── README.md
│   ├── best_model/
│   │   └── ...
│   └── checkpoints/
│       ├── checkpoint-step-500/
│       └── checkpoint-step-1000/
```

## 🤝 Integration with Deployment

After fine-tuning, your model is ready for deployment:

```bash
# Test the fine-tuned model
python scripts/test_model.py \
    --model-path ./fine_tuned_models/chat_chat_data/final_model

# Prepare for deployment
python scripts/prepare_for_deployment.py \
    --model-path ./fine_tuned_models/chat_chat_data/final_model \
    --output-dir ../07_deployment/models
```

## 📚 Additional Resources

- [Fine-tuning Best Practices](./docs/best_practices.md)
- [Dataset Preparation Guide](./docs/dataset_preparation.md)
- [LoRA Technical Details](./docs/lora_details.md)
- [Evaluation Metrics Guide](./docs/metrics_guide.md)

## 🎉 Next Steps

Once your model is fine-tuned:
1. Evaluate on test set
2. Run inference tests
3. Prepare for deployment (see `07_deployment`)
4. Monitor model performance in production

---

For questions or issues, please check the [FAQ](./docs/FAQ.md) or open an issue in the repository. 