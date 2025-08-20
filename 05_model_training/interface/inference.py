#!/usr/bin/env python3
"""
Interactive Inference Script for Hugging Face Transformers Model
===============================================================

This script provides an interactive interface to:
1. Choose from locally available checkpoints
2. Download checkpoints from Google Cloud Storage bucket
3. Perform text generation inference

Usage: python inference.py
"""

import torch
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage
from google.oauth2 import service_account

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Google Cloud Storage settings
BUCKET_NAME = "refocused-ai"
BUCKET_CHECKPOINT_PATH = "Checkpoints"

# Automatically detect the best available device (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default prompt for text generation
DEFAULT_PROMPT = "Hello, I am a language model,"

# ============================================================================
# CREDENTIAL SETUP AND VALIDATION
# ============================================================================

def get_storage_client(credentials_path: str | None = None, project_id: str | None = None):
    """Construct a GCS client; do not rely on env vars."""
    if credentials_path and os.path.exists(credentials_path):
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(project=project_id, credentials=creds)
    return storage.Client()

# ============================================================================
# SCRIPT INITIALIZATION
# ============================================================================

print("=" * 70)
print("🚀 ReFocused-AI Interactive Model Inference Script")
print("=" * 70)
print(f"🖥️  Device: {DEVICE}")
print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 CUDA Device: {torch.cuda.get_device_name()}")

# Early credential check to provide helpful feedback
print("\n🔐 Optional: Provide path to GCS credentials (press Enter to skip)")
gcs_key_path = input("Credentials JSON path (or blank): ").strip()
gcp_project_id = None
if gcs_key_path:
    gcp_project_id = input("GCP Project ID (optional): ").strip() or None
print("✅ Proceeding with", "provided credentials" if gcs_key_path else "default GCS auth (may be disabled)")

print("=" * 70)

# ============================================================================
# CHECKPOINT DISCOVERY AND SELECTION
# ============================================================================

def find_local_checkpoints():
    """Find all local checkpoint directories"""
    checkpoint_dirs = []
    
    # Common checkpoint locations in the project
    search_paths = [
        "05_model_training/checkpoints",
        "checkpoints", 
        "models",
        "05_model_training/cache",
        "."
    ]
    
    print("🔍 Scanning for local checkpoints...")
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for item in os.listdir(search_path):
                item_path = os.path.join(search_path, item)
                if os.path.isdir(item_path):
                    # Check if directory contains model files
                    model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
                    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
                    
                    has_model = any(os.path.exists(os.path.join(item_path, f)) for f in model_files)
                    has_tokenizer = any(os.path.exists(os.path.join(item_path, f)) for f in tokenizer_files)
                    
                    if has_model or has_tokenizer:
                        # Get relative path from project root
                        rel_path = os.path.relpath(item_path)
                        checkpoint_dirs.append({
                            'name': item,
                            'path': rel_path,
                            'has_model': has_model,
                            'has_tokenizer': has_tokenizer,
                            'size_mb': get_dir_size_mb(item_path)
                        })
    
    return checkpoint_dirs

def get_dir_size_mb(path):
    """Calculate directory size in MB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)
    except:
        return 0

def list_bucket_checkpoints():
    """List available checkpoints in the GCS bucket"""
    try:
        print("☁️  Connecting to Google Cloud Storage...")
        client = get_storage_client(gcs_key_path, gcp_project_id)
        bucket = client.bucket(BUCKET_NAME)
        
        print(f"📡 Scanning gs://{BUCKET_NAME}/{BUCKET_CHECKPOINT_PATH}/")
        blobs = list(bucket.list_blobs(prefix=f"{BUCKET_CHECKPOINT_PATH}/"))
        
        # Find unique checkpoint names
        checkpoints = set()
        for blob in blobs:
            # Extract checkpoint name from path
            path_parts = blob.name.split('/')
            if len(path_parts) >= 2:
                checkpoint_name = path_parts[1]
                if checkpoint_name and not checkpoint_name.endswith('.tar.gz'):
                    checkpoints.add(checkpoint_name)
                elif checkpoint_name.endswith('.tar.gz'):
                    checkpoints.add(checkpoint_name.replace('.tar.gz', ''))
        
        return sorted(list(checkpoints))
        
    except Exception as e:
        print(f"⚠️  Could not access bucket: {e}")
        return []

def download_checkpoint_from_bucket(checkpoint_name, download_dir="./downloaded_checkpoints"):
    """
    Download a checkpoint from GCS. If it's a training checkpoint (missing config.json),
    it will be automatically converted to an inference-ready format.
    """
    print(f"📥 Preparing checkpoint '{checkpoint_name}' from bucket...")

    # Check if credentials are available
    # Build client using provided credentials if any
    client = get_storage_client(gcs_key_path, gcp_project_id)

    os.makedirs(download_dir, exist_ok=True)
    training_path = os.path.join(download_dir, checkpoint_name)

    # Only download if the training checkpoint doesn't already exist
    if not os.path.exists(training_path):
        try:
            bucket = client.bucket(BUCKET_NAME)
            tar_blob_name = f"{BUCKET_CHECKPOINT_PATH}/{checkpoint_name}.tar.gz"
            tar_blob = bucket.blob(tar_blob_name)

            if tar_blob.exists():
                tar_path = f"{training_path}.tar.gz"
                print(f"  -> Downloading compressed checkpoint...")
                tar_blob.download_to_filename(tar_path)

                print(f"  -> Extracting checkpoint to '{training_path}'...")
                subprocess.run(["tar", "xzf", tar_path, "-C", download_dir], check=True)
                os.remove(tar_path) # Clean up the tarball
            else:
                # Add logic for non-tarball downloads if needed, for now we assume tar.gz
                print(f"❌ Compressed checkpoint '{tar_blob_name}' not found in bucket.")
                return None

        except Exception as e:
            print(f"❌ Error during download/extraction: {e}")
            return None
    else:
        print(f"✅ Found existing downloaded training checkpoint at '{training_path}'")

    # --- NEW: CONVERSION LOGIC ---
    # Check if the downloaded checkpoint needs to be converted
    config_path = os.path.join(training_path, "config.json")
    if not os.path.exists(config_path):
        print("\n🔄 Training checkpoint detected. Converting to inference format...")
        inference_path = os.path.join("./inference_ready_checkpoints", checkpoint_name)

        try:
            os.makedirs(inference_path, exist_ok=True)
            BASE_MODEL_NAME = "EleutherAI/gpt-neox-20b"

            print("  -> Loading base config and tokenizer...")
            config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

            print("  -> Overwriting config with your 1.2B model's architecture...")
            config.hidden_size = 2048
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.intermediate_size = 8192
            config.vocab_size = 50257

            print("  -> Loading your trained weights into the model structure...")
            model = AutoModelForCausalLM.from_pretrained(training_path, config=config)

            print(f"  -> Saving new inference-ready checkpoint to '{inference_path}'...")
            model.save_pretrained(inference_path)
            tokenizer.save_pretrained(inference_path)

            print(f"✅ Successfully converted. Using new path: {inference_path}")
            return inference_path # IMPORTANT: Return the path to the NEW checkpoint
        except Exception as e:
            print(f"❌ Fatal error during conversion: {e}")
            return None
    else:
        print("✅ Inference-ready checkpoint detected. No conversion needed.")
        return training_path # It was already in the right format

def select_checkpoint():
    """Interactive checkpoint selection"""
    print("\n" + "=" * 50)
    print("📁 CHECKPOINT SELECTION")
    print("=" * 50)
    
    # Find local checkpoints
    local_checkpoints = find_local_checkpoints()
    
    if local_checkpoints:
        print(f"\n✅ Found {len(local_checkpoints)} local checkpoint(s):")
        for i, cp in enumerate(local_checkpoints, 1):
            status = "🔥" if cp['has_model'] and cp['has_tokenizer'] else "⚠️ "
            print(f"  {i}. {cp['name']} ({cp['size_mb']:.1f} MB) {status}")
            print(f"     📂 {cp['path']}")
    else:
        print("📭 No local checkpoints found")
    
    # Check bucket checkpoints
    bucket_checkpoints = list_bucket_checkpoints()
    if bucket_checkpoints:
        print(f"\n☁️  Found {len(bucket_checkpoints)} checkpoint(s) in bucket:")
        for cp in bucket_checkpoints[:10]:  # Show first 10
            print(f"  • {cp}")
        if len(bucket_checkpoints) > 10:
            print(f"  ... and {len(bucket_checkpoints) - 10} more")
    
    # Selection menu
    print(f"\n🎯 SELECT AN OPTION:")
    if local_checkpoints:
        print(f"  1-{len(local_checkpoints)}: Use local checkpoint")
    if bucket_checkpoints:
        print(f"  B: Download from bucket")
    print(f"  M: Manual path entry")
    print(f"  Q: Quit")
    
    while True:
        choice = input("\n👉 Your choice: ").strip().upper()
        
        if choice == 'Q':
            print("👋 Goodbye!")
            sys.exit(0)
        
        elif choice == 'M':
            manual_path = input("📂 Enter checkpoint path: ").strip()
            if os.path.exists(manual_path):
                return manual_path
            else:
                print(f"❌ Path not found: {manual_path}")
                continue
        
        elif choice == 'B' and bucket_checkpoints:
            print(f"\n☁️  Available checkpoints in bucket:")
            for i, cp in enumerate(bucket_checkpoints, 1):
                print(f"  {i}. {cp}")
            
            try:
                bucket_choice = int(input("👉 Select checkpoint number: ")) - 1
                if 0 <= bucket_choice < len(bucket_checkpoints):
                    selected_checkpoint = bucket_checkpoints[bucket_choice]
                    downloaded_path = download_checkpoint_from_bucket(selected_checkpoint)
                    if downloaded_path:
                        return downloaded_path
                else:
                    print("❌ Invalid selection")
                    continue
            except ValueError:
                print("❌ Please enter a valid number")
                continue
        
        elif choice.isdigit() and local_checkpoints:
            try:
                local_choice = int(choice) - 1
                if 0 <= local_choice < len(local_checkpoints):
                    return local_checkpoints[local_choice]['path']
                else:
                    print("❌ Invalid selection")
                    continue
            except ValueError:
                print("❌ Please enter a valid number")
                continue
        
        else:
            print("❌ Invalid choice. Try again.")

# Get checkpoint path
CHECKPOINT_PATH = select_checkpoint()
print(f"\n🎯 Selected checkpoint: {CHECKPOINT_PATH}")

# Get custom prompt
print(f"\n💭 PROMPT CONFIGURATION")
print(f"Default prompt: '{DEFAULT_PROMPT}'")
custom_prompt = input("Enter custom prompt (or press Enter for default): ").strip()
PROMPT_TEXT = custom_prompt if custom_prompt else DEFAULT_PROMPT

print("=" * 70)

# ============================================================================
# LOAD TOKENIZER AND MODEL SECTION
# ============================================================================

try:
    print("\n🔄 Loading tokenizer...")
    # Load the tokenizer from the checkpoint directory
    # The tokenizer handles text preprocessing and postprocessing
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    print("✅ Tokenizer loaded successfully.")
    
    print("\n🔄 Loading model... (This might take a few minutes and significant RAM/VRAM)")
    # Load the model from the checkpoint directory
    # Consider uncommenting the options below for memory optimization:
    # torch_dtype=torch.float16  # Use half precision to reduce VRAM usage
    # low_cpu_mem_usage=True     # Reduce RAM usage during loading
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        # torch_dtype=torch.float16,  # Uncomment for VRAM optimization
        # low_cpu_mem_usage=True      # Uncomment for RAM optimization
    )
    
    # Move model to the appropriate device (GPU or CPU)
    print(f"🔄 Moving model to {DEVICE}...")
    model.to(DEVICE)
    
    # Set model to evaluation mode - this disables dropout and batch normalization
    # layers that behave differently during training vs inference
    model.eval()
    print("✅ Model loaded successfully and moved to device.")
    
except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")
    print("\n🔍 Please check the following:")
    print(f"   • CHECKPOINT_PATH is correct: {CHECKPOINT_PATH}")
    print("   • Directory exists and contains required files:")
    print("     - pytorch_model.bin (or model.safetensors)")
    print("     - config.json")
    print("     - tokenizer.json")
    print("     - tokenizer_config.json")
    print("   • You have sufficient RAM/VRAM available")
    print("   • All required dependencies are installed")
    sys.exit(1)

# ============================================================================
# GENERATE TEXT SECTION
# ============================================================================

try:
    print(f"\n🚀 Generating text for prompt: '{PROMPT_TEXT}'")
    print("⏳ Please wait while the model generates text...")
    
    # Encode the input prompt into tokens that the model can understand
    # return_tensors="pt" returns PyTorch tensors
    input_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").to(DEVICE)
    
    # Use torch.no_grad() to disable gradient computation during inference
    # This saves memory and speeds up computation since we're not training
    with torch.no_grad():
        # Generate text using the model with various sampling parameters
        outputs = model.generate(
            input_ids.input_ids,           # Input token IDs
            max_length=100,                # Maximum length of generated sequence
            num_return_sequences=1,        # Number of different sequences to generate
            do_sample=True,                # Enable sampling (vs greedy decoding)
            temperature=0.7,               # Controls randomness (lower = more focused)
            top_k=50,                      # Consider only top k most likely next tokens
            top_p=0.95,                    # Nucleus sampling: consider tokens with cumulative probability <= top_p
            pad_token_id=tokenizer.eos_token_id  # Use end-of-sequence token for padding
        )
    
    # Decode the generated tokens back to human-readable text
    # skip_special_tokens=True removes tokens like [PAD], [EOS], etc.
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display the results clearly
    print("\n" + "=" * 70)
    print("--- MODEL OUTPUT ---")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)
    print("--- END OF OUTPUT ---")
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Error during text generation: {e}")
    print("This could be due to:")
    print("   • Insufficient GPU/CPU memory")
    print("   • Model compatibility issues")
    print("   • Invalid generation parameters")
    sys.exit(1)

# ============================================================================
# END MESSAGE
# ============================================================================

print("\n✅ Inference script finished.")
print("💡 To generate different text, run the script again and try different prompts.")
print("💡 To adjust generation quality, experiment with temperature, top_k, and top_p parameters.")
print("🚀 WOOOOOOOOOOO!") 
