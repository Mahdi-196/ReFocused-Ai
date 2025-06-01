#!/bin/bash
# Configure accelerate for FSDP training

echo "Configuring Accelerate for FSDP training..."

# Copy our config to the default location
mkdir -p ~/.cache/huggingface/accelerate
cp accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

echo "Accelerate configured successfully!"
echo "You can now use 'accelerate launch' with FSDP enabled by default." 