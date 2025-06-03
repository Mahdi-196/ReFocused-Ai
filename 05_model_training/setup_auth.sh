#!/bin/bash

# Google Cloud Authentication Setup
# Sets environment variables for authenticated GCS access

# Path to your service account key file
export GOOGLE_APPLICATION_CREDENTIALS="./credentials/black-dragon-461023-t5-93452a49f86b.json"

# Your Google Cloud project ID
export GOOGLE_CLOUD_PROJECT="black-dragon-461023-t5"

# Verify the setup
echo "🔐 Google Cloud Authentication Setup"
echo "   Project: $GOOGLE_CLOUD_PROJECT"
echo "   Credentials: $GOOGLE_APPLICATION_CREDENTIALS"

# Test authentication
if command -v gcloud &> /dev/null; then
    echo "🧪 Testing authentication..."
    gcloud auth application-default print-access-token > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Authentication successful!"
    else
        echo "⚠️  Authentication test failed - make sure your key file exists"
    fi
else
    echo "ℹ️  gcloud CLI not found - authentication will be tested during training"
fi

echo ""
echo "🚀 Ready to run training with authenticated GCS access"
echo "   Bucket: refocused-ai"
echo "   Checkpoint path: Checkpoints/" 