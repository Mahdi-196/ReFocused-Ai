#!/usr/bin/env python3
"""
Tokenizer Comparison and Examples Script
=======================================

Demonstrates different tokenizer types with practical examples.
Shows why different tokenizers are used for different models and domains.

Requirements:
- tokenizers
- transformers
- tabulate

Author: AI Assistant
Version: 1.0
"""

import time
from typing import List, Dict, Any
from pathlib import Path

try:
    from tokenizers import (
        Tokenizer, ByteLevelBPETokenizer,
        models, normalizers, pre_tokenizers, processors, trainers
    )
    from transformers import (
        AutoTokenizer, GPT2TokenizerFast, BertTokenizerFast, 
        XLNetTokenizerFast, T5TokenizerFast
    )
    from tabulate import tabulate
    print("✓ All required packages are available")
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Please install missing packages:")
    print("pip install tokenizers transformers tabulate")
    exit(1)


class TokenizerComparison:
    """Comprehensive tokenizer comparison and demonstration."""
    
    def __init__(self):
        self.test_texts = [
            "I will build better habits.",
            "The quick brown fox jumps over the lazy dog.",
            "Tokenization is fundamental to natural language processing.",
            "машинное обучение",  # Russian: machine learning
            "机器学习",  # Chinese: machine learning
            "hypertension cardiovascular pharmaceutical",  # Medical terms
            "def tokenize_text(input_string): return tokens",  # Code
            "COVID-19 mRNA vaccination immunocompromised",  # Recent terms
            "Hello, world! How are you doing today? 😊",  # Emoji and punctuation
            "The state-of-the-art transformer-based architecture.",  # Hyphenated terms
        ]
        
        self.popular_tokenizers = {}
        self.custom_tokenizers = {}
    
    def load_popular_tokenizers(self):
        """Load popular pre-trained tokenizers for comparison."""
        print("🔧 Loading popular tokenizers...")
        
        tokenizer_configs = {
            "GPT-2 (BPE)": "gpt2",
            "BERT (WordPiece)": "bert-base-uncased", 
            "RoBERTa (BPE)": "roberta-base",
            "T5 (SentencePiece)": "t5-small",
            "XLNet (Unigram)": "xlnet-base-cased"
        }
        
        for name, model_name in tokenizer_configs.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.popular_tokenizers[name] = tokenizer
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    def create_bpe_tokenizer(self, vocab_size: int = 8000) -> ByteLevelBPETokenizer:
        """Create a custom BPE tokenizer for demonstration."""
        print(f"🔨 Creating custom BPE tokenizer (vocab: {vocab_size})...")
        
        tokenizer = ByteLevelBPETokenizer()
        
        # Create temporary training data
        temp_file = Path("temp_training_data.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            # Repeat test texts to create training corpus
            for _ in range(100):
                for text in self.test_texts:
                    f.write(text + '\n')
        
        # Train tokenizer
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        tokenizer.train(
            files=[str(temp_file)],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=False
        )
        
        # Cleanup
        temp_file.unlink()
        
        return tokenizer
    
    def create_wordpiece_tokenizer(self, vocab_size: int = 8000) -> Tokenizer:
        """Create a custom WordPiece tokenizer from scratch."""
        print(f"🔨 Creating custom WordPiece tokenizer (vocab: {vocab_size})...")
        
        # Initialize WordPiece model
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        
        # BERT-style normalization
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        
        # BERT-style pre-tokenization  
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        
        # Create training data
        temp_file = Path("temp_wordpiece_data.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for _ in range(100):
                for text in self.test_texts:
                    f.write(text + '\n')
        
        # Train with WordPiece trainer
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        
        tokenizer.train([str(temp_file)], trainer)
        
        # Add BERT-style post-processing
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]"))
            ]
        )
        
        # Cleanup
        temp_file.unlink()
        
        return tokenizer
    
    def create_unigram_tokenizer(self, vocab_size: int = 8000) -> Tokenizer:
        """Create a custom Unigram tokenizer from scratch."""
        print(f"🔨 Creating custom Unigram tokenizer (vocab: {vocab_size})...")
        
        # Initialize Unigram model
        tokenizer = Tokenizer(models.Unigram())
        
        # XLNet-style normalization
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'), 
            normalizers.NFKD(),
            normalizers.StripAccents()
        ])
        
        # Metaspace pre-tokenization (SentencePiece style)
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        
        # Create training data
        temp_file = Path("temp_unigram_data.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for _ in range(100):
                for text in self.test_texts:
                    f.write(text + '\n')
        
        # Train with Unigram trainer
        special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token="<unk>"
        )
        
        tokenizer.train([str(temp_file)], trainer)
        
        # Cleanup
        temp_file.unlink()
        
        return tokenizer
    
    def analyze_tokenizer(self, tokenizer, name: str, text: str) -> Dict[str, Any]:
        """Analyze a tokenizer's performance on given text."""
        
        start_time = time.time()
        
        # Handle different tokenizer types
        if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
            # Transformers tokenizer
            if hasattr(tokenizer, 'tokenize'):
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            else:
                # Fallback for some tokenizers
                token_ids = tokenizer.encode(text)
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
                decoded = tokenizer.decode(token_ids)
        else:
            # Hugging Face tokenizers library
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            token_ids = encoding.ids
            decoded = tokenizer.decode(token_ids)
        
        tokenization_time = time.time() - start_time
        
        # Calculate metrics
        char_count = len(text)
        token_count = len(tokens)
        compression_ratio = char_count / max(token_count, 1)
        
        # Count unknown tokens
        unk_tokens = 0
        unk_patterns = ["<unk>", "[UNK]", "▁", "<|endoftext|>"]
        for token in tokens:
            if any(unk in str(token) for unk in unk_patterns):
                unk_tokens += 1
        
        unknown_rate = unk_tokens / max(token_count, 1)
        
        return {
            'tokenizer': name,
            'text': text[:50] + "..." if len(text) > 50 else text,
            'tokens': tokens,
            'token_ids': token_ids,
            'decoded': decoded,
            'char_count': char_count,
            'token_count': token_count,
            'compression_ratio': compression_ratio,
            'unknown_rate': unknown_rate,
            'tokenization_time': tokenization_time * 1000  # milliseconds
        }
    
    def compare_tokenizers_on_text(self, text: str):
        """Compare all tokenizers on a single text."""
        print(f"\n📝 Comparing tokenizers on: '{text}'")
        print("=" * 80)
        
        results = []
        
        # Test popular tokenizers
        for name, tokenizer in self.popular_tokenizers.items():
            try:
                result = self.analyze_tokenizer(tokenizer, name, text)
                results.append(result)
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        # Test custom tokenizers
        for name, tokenizer in self.custom_tokenizers.items():
            try:
                result = self.analyze_tokenizer(tokenizer, name, text)
                results.append(result)
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        # Create comparison table
        table_data = []
        for result in results:
            tokens_str = str(result['tokens'][:8])  # First 8 tokens
            if len(result['tokens']) > 8:
                tokens_str = tokens_str[:-1] + ", ...]"
            
            table_data.append([
                result['tokenizer'],
                result['token_count'],
                f"{result['compression_ratio']:.2f}",
                f"{result['unknown_rate']*100:.1f}%",
                f"{result['tokenization_time']:.2f}ms",
                tokens_str
            ])
        
        headers = ["Tokenizer", "Tokens", "Compression", "Unknown%", "Time", "Sample Tokens"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def detailed_token_analysis(self, text: str):
        """Show detailed token-by-token comparison."""
        print(f"\n🔍 Detailed Token Analysis: '{text}'")
        print("=" * 80)
        
        for name, tokenizer in {**self.popular_tokenizers, **self.custom_tokenizers}.items():
            try:
                result = self.analyze_tokenizer(tokenizer, name, text)
                print(f"\n{name}:")
                print(f"  Tokens: {result['tokens']}")
                print(f"  Token IDs: {result['token_ids']}")
                print(f"  Decoded: '{result['decoded']}'")
                print(f"  Stats: {result['token_count']} tokens, {result['compression_ratio']:.2f} compression")
            except Exception as e:
                print(f"  Error: {e}")
    
    def domain_specific_analysis(self):
        """Analyze tokenizer performance on domain-specific texts."""
        print("\n🎯 Domain-Specific Analysis")
        print("=" * 80)
        
        domain_texts = {
            "Medical": "hypertension cardiovascular pharmaceutical immunocompromised",
            "Code": "def tokenize_text(input_string): return tokens.split()",
            "Multilingual": "машинное обучение 机器学习 apprentissage automatique",
            "Recent Terms": "COVID-19 mRNA SARS-CoV-2 cryptocurrency blockchain",
            "Social Media": "OMG this is so cool! 😊 #AI #MachineLearning @username",
            "Scientific": "polymerase chain reaction nucleotide sequences genomics"
        }
        
        for domain, text in domain_texts.items():
            print(f"\n📊 {domain} Domain:")
            self.compare_tokenizers_on_text(text)
    
    def performance_benchmark(self):
        """Benchmark tokenizer performance on different text lengths."""
        print("\n⚡ Performance Benchmark")
        print("=" * 80)
        
        # Create texts of different lengths
        test_sizes = {
            "Short (50 chars)": "The quick brown fox jumps over the lazy dog.",
            "Medium (200 chars)": " ".join(self.test_texts[:3]),
            "Long (500 chars)": " ".join(self.test_texts * 2),
        }
        
        for size_name, text in test_sizes.items():
            print(f"\n📏 {size_name} ({len(text)} chars):")
            
            perf_data = []
            for name, tokenizer in {**self.popular_tokenizers, **self.custom_tokenizers}.items():
                try:
                    # Time multiple runs
                    times = []
                    for _ in range(10):
                        start = time.time()
                        self.analyze_tokenizer(tokenizer, name, text)
                        times.append((time.time() - start) * 1000)
                    
                    avg_time = sum(times) / len(times)
                    
                    # Get basic stats
                    result = self.analyze_tokenizer(tokenizer, name, text)
                    
                    perf_data.append([
                        name,
                        result['token_count'],
                        f"{result['compression_ratio']:.2f}",
                        f"{avg_time:.2f}ms"
                    ])
                except Exception as e:
                    perf_data.append([name, "Error", "Error", f"Error: {e}"])
            
            headers = ["Tokenizer", "Token Count", "Compression", "Avg Time"]
            print(tabulate(perf_data, headers=headers, tablefmt="grid"))
    
    def vocabulary_analysis(self):
        """Analyze vocabulary characteristics of different tokenizers."""
        print("\n📚 Vocabulary Analysis")
        print("=" * 80)
        
        vocab_data = []
        
        for name, tokenizer in {**self.popular_tokenizers, **self.custom_tokenizers}.items():
            try:
                if hasattr(tokenizer, 'vocab_size'):
                    vocab_size = tokenizer.vocab_size
                elif hasattr(tokenizer, 'get_vocab_size'):
                    vocab_size = tokenizer.get_vocab_size()
                elif hasattr(tokenizer, 'vocab'):
                    vocab_size = len(tokenizer.vocab)
                else:
                    vocab_size = "Unknown"
                
                # Try to get special tokens
                special_tokens = []
                if hasattr(tokenizer, 'special_tokens_map'):
                    special_tokens = list(tokenizer.special_tokens_map.values())
                elif hasattr(tokenizer, 'added_tokens_decoder'):
                    special_tokens = list(tokenizer.added_tokens_decoder.values())[:5]
                
                vocab_data.append([
                    name,
                    vocab_size,
                    len(special_tokens),
                    str(special_tokens[:3]) if special_tokens else "None"
                ])
            except Exception as e:
                vocab_data.append([name, "Error", "Error", str(e)[:30]])
        
        headers = ["Tokenizer", "Vocab Size", "Special Tokens", "Sample Special Tokens"]
        print(tabulate(vocab_data, headers=headers, tablefmt="grid"))
    
    def run_comprehensive_comparison(self):
        """Run the complete tokenizer comparison analysis."""
        print("🎯 COMPREHENSIVE TOKENIZER COMPARISON")
        print("=" * 80)
        print("This analysis compares different tokenizer types and their characteristics.")
        
        # Load tokenizers
        self.load_popular_tokenizers()
        
        # Create custom tokenizers for comparison
        print("\n🔧 Creating custom tokenizers for comparison...")
        self.custom_tokenizers["Custom BPE"] = self.create_bpe_tokenizer(8000)
        self.custom_tokenizers["Custom WordPiece"] = self.create_wordpiece_tokenizer(8000)
        self.custom_tokenizers["Custom Unigram"] = self.create_unigram_tokenizer(8000)
        
        # Run analyses
        self.vocabulary_analysis()
        
        # Compare on standard texts
        for text in self.test_texts[:3]:  # First 3 test texts
            self.compare_tokenizers_on_text(text)
        
        # Domain-specific analysis
        self.domain_specific_analysis()
        
        # Performance benchmark
        self.performance_benchmark()
        
        # Detailed analysis on interesting text
        self.detailed_token_analysis("The state-of-the-art transformer-based architecture uses subword tokenization.")
        
        print("\n" + "=" * 80)
        print("🎉 COMPARISON COMPLETE!")
        print("\n📊 KEY TAKEAWAYS:")
        print("• BPE (GPT-2): Good for general text, handles OOV well")
        print("• WordPiece (BERT): Optimal for masked language modeling")
        print("• Unigram (XLNet): Theoretically optimal, good for agglutinative languages")
        print("• Custom tokenizers: Domain-specific optimization possible")
        print("• Choice depends on: model architecture, domain, language, performance needs")


def main():
    """Main function to run tokenizer comparison."""
    comparison = TokenizerComparison()
    comparison.run_comprehensive_comparison()


if __name__ == "__main__":
    main() 