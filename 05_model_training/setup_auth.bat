@echo off

REM Google Cloud Authentication Setup for Windows
REM Sets environment variables for authenticated GCS access

REM Path to your service account key file
set GOOGLE_APPLICATION_CREDENTIALS=./credentials/black-dragon-461023-t5-93452a49f86b.json

REM Your Google Cloud project ID
set GOOGLE_CLOUD_PROJECT=black-dragon-461023-t5

REM Verify the setup
echo 🔐 Google Cloud Authentication Setup
echo    Project: %GOOGLE_CLOUD_PROJECT%
echo    Credentials: %GOOGLE_APPLICATION_CREDENTIALS%

echo.
echo 🚀 Ready to run training with authenticated GCS access
echo    Bucket: refocused-ai
echo    Checkpoint path: Checkpoints/ 