# 🚀 ReFocused-AI: 1.2B Parameter Language Model

Complete training system for the ReFocused-AI language model with Google Cloud Storage integration and authenticated checkpoint uploading.

## 🌱 Why this project exists

I built this to power my ReFocused application. It generates useful, personalized content like recommendations, interesting facts, weekly themes, and it supports chatting with the model. I also wanted to see if I could build a full end‑to‑end training stack after reading projects like llm.c — the low‑level details were fascinating and I wanted to learn by doing.

## 📚 What I learned

I ended up going a lot deeper than I planned, mostly because the internals kept pulling me in even after everything “worked.” Getting from raw text to tokens to attention blocks made me appreciate how many tiny choices (padding, masks, precision) quietly decide whether training feels smooth or fragile. I also built a practical feel for the whole pipeline: data quality matters most, stability and checkpointing aren’t optional, and scaling only helps if your I/O keeps up.

I’m still kind of amazed that these models stay coherent and are often accurate in normal use — and I’ve also seen enough failure cases to respect their limits. On the engineering side, the unglamorous parts carried a lot of weight: sharding, streaming, background uploads, and “resume actually resumes.” The performance wins that consistently helped were simple: mixed precision, `torch.compile` where it works, and sensible dataloader settings.

## 💛 Favroite feature

my favorite things about this pipeline is it really is user friendly it give steps tells you what worked and what didnt and how to fix it all in the command line another thing for me is i hate emojis in applications i think its a lower quality look but in this theyre spammed throughtout just to give that terminal some color plus seeing the rocket emoji after fialing over and over or the green checkmark after a bunch of red x's is enough to bring a tear down your face.

## 🧱 Pipeline overview (end‑to‑end)

1. `01_data_collection/`: Optional collectors for Reddit/Wikipedia; real‑time monitoring tools.
2. `02_data_processing/`: Clean, dedupe, score quality, and create train/val/test splits.
3. `03_tokenizer_training/`: Train a ByteLevel BPE tokenizer (GPT‑2 style) on your cleaned text.
4. `04_data_tokenization/`: Convert text into fixed‑length token sequences (`.npz` with input_ids/masks).
5. `05_model_training/`: Train the 1.2B GPT‑NeoX model; mixed precision, resume, and GCS checkpoints.
6. `07_utilities/`: Analysis scripts (dataset stats, tokenized data checks, quick counts).

## 📋 Quick Start

**👉 For complete setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### TL;DR Setup
```bash
# Clone and setup
git clone https://github.com/Mahdi-196/ReFocused-Ai.git
cd ReFocused-Ai
python -m venv refocused_env
source refocused_env/bin/activate  # Linux/Mac
# or refocused_env\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure training
cd 05_model_training
# [Add your service account key to credentials/]
./start_training.sh --config test
```

## 🗂️ Project Structure

```
ReFocused-Ai/
├── 01_data_collection/         # Collect + monitor (Reddit/Wikipedia)
├── 02_data_processing/         # Clean, quality‑score, splits
├── 03_tokenizer_training/      # Train tokenizer (BPE)
├── 04_data_tokenization/       # Produce .npz token shards
├── 05_model_training/          # 🎯 Training system (GPT‑NeoX 1.2B)
│   ├── train.py                # Main training script
│   ├── start_training.sh       # One‑click launcher
│   ├── TRAINING_README.md      # Detailed training docs
│   ├── configs/                # Model/training configs
│   ├── utils/                  # Dataloaders, checkpoints, metrics
│   ├── credentials/            # GCS keys (user‑provided)
│   └── checkpoints/            # Local checkpoint storage
├── 07_utilities/               # Analysis scripts
├── SETUP_GUIDE.md              # 📖 Complete setup guide
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ✨ Key Features (whole pipeline)

- **Data collection**: Real‑time Reddit/Wikipedia collectors with monitoring (optional)
- **Processing**: Dedup/quality filters, clean JSONL, balanced splits
- **Tokenizer**: Train BPE (GPT‑2 style) and save Transformers‑compatible artifacts
- **Tokenization**: Shard to `.npz` with input_ids/attention_mask ready for PyTorch
- **Training**: Mixed precision, device‑aware `torch.compile`, resume, metrics
- **Checkpoints**: Background uploads to GCS with metadata; reliable resume
- **Monitoring**: CLI logs, TensorBoard support; progress and performance hints
- **Configs**: Test and production presets; easy overrides for steps/batch size

## 🚀 End‑to‑end Quick Start

```bash
# 1) (Optional) Collect data
python 01_data_collection/Wikipedia-Collector.py  # or use existing data

# 2) Process & split
python 02_data_processing/data_cleaner.py
python 02_data_processing/data_processor.py

# 3) Train tokenizer (or skip if using an existing one)
python 03_tokenizer_training/train_tokenizer.py

# 4) Tokenize to .npz shards
python 04_data_tokenization/run_full_tokenization.py

# 5) Train the model
./start_training.sh --config test

# Production training
./start_training.sh --config production

# Custom steps
./start_training.sh --config test --max-steps 2000

# Resume from checkpoint
./start_training.sh --config test --resume checkpoint-name

# Monitor progress
tail -f logs/training.log
```

## 📊 Training Configurations

| Config | Steps | Files | Batch Size | Duration | Purpose |
|--------|-------|-------|------------|----------|---------|
| **test** | 1000 | 5 | 1 | ~10-30 min | Testing, experiments |
| **production** | 10000 | All | 4 | Hours-days | Full training |

## ☁️ Checkpoint System

- **Automatic uploads** to `gs://refocused-ai/Checkpoints/`
- **Comprehensive metadata** with training metrics
- **Background processing** for non-blocking uploads
- **Resume capability** from any checkpoint
- **Local cleanup** of old checkpoints

## 🔧 Requirements

- **Python 3.9+** (recommended: 3.11)
- **PyTorch 2.0+** with CUDA support (optional but recommended)
- **Google Cloud Storage** access with service account credentials
- **8GB+ RAM** (16GB+ recommended)
- **NVIDIA GPU** (optional but significantly faster)

## 📖 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Complete setup instructions with virtual environment
- **[05_model_training/TRAINING_README.md](05_model_training/TRAINING_README.md)**: Detailed training documentation
- **[configs/](05_model_training/configs/)**: Training and model configuration files

## 🎯 Model Details

- **Architecture**: GPT-NeoX
- **Parameters**: ~1.2 billion
- **Context Length**: 2048 tokens  
- **Vocabulary**: 50,257 tokens
- **Training Data**: Reddit conversations (cleaned and tokenized)

## 🤝 Contributing

1. Follow the setup guide to get training working
2. Make changes in appropriate directories
3. Test with `./start_training.sh --config test --max-steps 5`
4. Submit pull requests with clear descriptions

## 📞 Support

For issues:
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) troubleshooting section
2. Review training logs in `05_model_training/logs/`
3. Verify virtual environment and dependencies
4. Ensure credentials are properly configured

---

**Ready to train? Start with [SETUP_GUIDE.md](SETUP_GUIDE.md)! 🚀** 