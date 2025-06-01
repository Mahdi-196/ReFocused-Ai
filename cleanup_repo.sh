#!/bin/bash

echo "🧹 Repository Cleanup Script"
echo "============================="
echo ""
echo "⚠️  WARNING: This will rewrite Git history!"
echo "⚠️  Make sure you have a backup of your repository!"
echo ""
echo "This script will:"
echo "1. Remove all .npz files from Git history"
echo "2. Remove data/training/ directory from Git history"
echo "3. Clean up the repository"
echo ""

read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "🔍 Checking repository size before cleanup..."
du -sh .git/

echo ""
echo "🧹 Removing .npz files from Git history..."
git filter-branch --force --index-filter \
    'git rm --cached --ignore-unmatch "*.npz"' \
    --prune-empty --tag-name-filter cat -- --all

echo ""
echo "🧹 Removing data/training/ directory from Git history..."
git filter-branch --force --index-filter \
    'git rm -r --cached --ignore-unmatch data/training/' \
    --prune-empty --tag-name-filter cat -- --all

echo ""
echo "🧹 Removing cache directories from Git history..."
git filter-branch --force --index-filter \
    'git rm -r --cached --ignore-unmatch 05_model_training/cache/ 05_model_training/preprocessed_cache/ preprocessed_cache/ cache/' \
    --prune-empty --tag-name-filter cat -- --all

echo ""
echo "🗑️  Cleaning up Git repository..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "📊 Repository size after cleanup:"
du -sh .git/

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Force push to update remote: git push --force-with-lease origin main"
echo "3. Inform collaborators to re-clone the repository"
echo ""
echo "⚠️  Note: All collaborators will need to re-clone the repository!" 