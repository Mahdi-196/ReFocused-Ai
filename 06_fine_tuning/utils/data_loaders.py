#!/usr/bin/env python3
"""
Data loaders for fine-tuning different tasks
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
import logging

logger = logging.getLogger(__name__)


class FineTuningDataset(Dataset):
    """Base dataset class for fine-tuning"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        task_type: str = "custom"
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.examples = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load data from file"""
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.examples = data
                else:
                    self.examples = [data]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        logger.info(f"Loaded {len(self.examples)} examples from {self.data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Process based on task type
        if self.task_type == "chat":
            text = self._format_chat(example)
        elif self.task_type == "instruct":
            text = self._format_instruct(example)
        elif self.task_type == "code":
            text = self._format_code(example)
        else:
            text = self._format_custom(example)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        labels = encoding["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    def _format_chat(self, example: Dict) -> str:
        """Format example for chat fine-tuning"""
        if "messages" in example:
            # Format as conversation
            conversation = ""
            for message in example["messages"]:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"<|{role}|>{content}<|end|>\n"
            return conversation
        elif "prompt" in example and "response" in example:
            return f"<|user|>{example['prompt']}<|end|>\n<|assistant|>{example['response']}<|end|>"
        else:
            return example.get("text", "")
    
    def _format_instruct(self, example: Dict) -> str:
        """Format example for instruction fine-tuning"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def _format_code(self, example: Dict) -> str:
        """Format example for code fine-tuning"""
        if "prompt" in example and "completion" in example:
            return f"# Task: {example['prompt']}\n\n{example['completion']}"
        elif "code" in example:
            return example["code"]
        else:
            return example.get("text", "")
    
    def _format_custom(self, example: Dict) -> str:
        """Format example for custom fine-tuning"""
        # Try common formats
        if "text" in example:
            return example["text"]
        elif "prompt" in example and "completion" in example:
            return f"{example['prompt']}\n{example['completion']}"
        elif "input" in example and "output" in example:
            return f"{example['input']}\n{example['output']}"
        else:
            # Concatenate all string values
            text_parts = []
            for key, value in example.items():
                if isinstance(value, str):
                    text_parts.append(value)
            return "\n".join(text_parts)


class HuggingFaceDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets"""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        task_type: str = "custom",
        split: str = "train",
        subset: Optional[str] = None,
        text_column: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        # Load HuggingFace dataset
        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=split)
        else:
            self.dataset = load_dataset(dataset_name, split=split)
        
        # Determine text column
        if text_column:
            self.text_column = text_column
        else:
            # Try to auto-detect
            possible_columns = ["text", "content", "prompt", "instruction", "input"]
            for col in possible_columns:
                if col in self.dataset.column_names:
                    self.text_column = col
                    break
            else:
                # Use first string column
                for col in self.dataset.column_names:
                    if isinstance(self.dataset[0][col], str):
                        self.text_column = col
                        break
        
        logger.info(f"Using text column: {self.text_column}")
        logger.info(f"Loaded {len(self.dataset)} examples from {dataset_name}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Get text based on task type
        if self.task_type == "chat" and "messages" in example:
            text = self._format_chat(example)
        elif self.task_type == "instruct" and all(k in example for k in ["instruction", "output"]):
            text = self._format_instruct(example)
        else:
            text = example[self.text_column]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        labels = encoding["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    def _format_chat(self, example: Dict) -> str:
        """Format chat example"""
        conversation = ""
        for message in example["messages"]:
            role = message.get("role", "user")
            content = message.get("content", "")
            conversation += f"<|{role}|>{content}<|end|>\n"
        return conversation
    
    def _format_instruct(self, example: Dict) -> str:
        """Format instruction example"""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def create_fine_tuning_dataloader(
    dataset_path: str,
    task_type: str,
    tokenizer: AutoTokenizer,
    config,
    accelerator,
    eval_split: float = 0.1
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for fine-tuning"""
    
    # Check if it's a HuggingFace dataset
    if "/" in dataset_path and not Path(dataset_path).exists():
        # Assume HuggingFace dataset
        logger.info(f"Loading HuggingFace dataset: {dataset_path}")
        
        # Create train dataset
        train_dataset = HuggingFaceDataset(
            dataset_name=dataset_path,
            tokenizer=tokenizer,
            max_length=config.max_length,
            task_type=task_type,
            split="train"
        )
        
        # Try to load validation set
        eval_dataset = None
        try:
            eval_dataset = HuggingFaceDataset(
                dataset_name=dataset_path,
                tokenizer=tokenizer,
                max_length=config.max_length,
                task_type=task_type,
                split="validation"
            )
        except:
            try:
                eval_dataset = HuggingFaceDataset(
                    dataset_name=dataset_path,
                    tokenizer=tokenizer,
                    max_length=config.max_length,
                    task_type=task_type,
                    split="test"
                )
            except:
                # Create eval split from train
                if eval_split > 0:
                    total_size = len(train_dataset)
                    eval_size = int(total_size * eval_split)
                    train_size = total_size - eval_size
                    
                    train_dataset, eval_dataset = torch.utils.data.random_split(
                        train_dataset, [train_size, eval_size]
                    )
    else:
        # Local dataset
        dataset_path = Path(dataset_path)
        
        if dataset_path.is_file():
            # Single file
            dataset = FineTuningDataset(
                data_path=dataset_path,
                tokenizer=tokenizer,
                max_length=config.max_length,
                task_type=task_type
            )
            
            # Split into train/eval
            if eval_split > 0:
                total_size = len(dataset)
                eval_size = int(total_size * eval_split)
                train_size = total_size - eval_size
                
                train_dataset, eval_dataset = torch.utils.data.random_split(
                    dataset, [train_size, eval_size]
                )
            else:
                train_dataset = dataset
                eval_dataset = None
        else:
            # Directory with train/eval splits
            train_path = dataset_path / "train.jsonl"
            eval_path = dataset_path / "eval.jsonl"
            
            if not train_path.exists():
                train_path = dataset_path / "train.json"
                eval_path = dataset_path / "eval.json"
            
            train_dataset = FineTuningDataset(
                data_path=train_path,
                tokenizer=tokenizer,
                max_length=config.max_length,
                task_type=task_type
            )
            
            if eval_path.exists():
                eval_dataset = FineTuningDataset(
                    data_path=eval_path,
                    tokenizer=tokenizer,
                    max_length=config.max_length,
                    task_type=task_type
                )
            else:
                eval_dataset = None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    return train_dataloader, eval_dataloader


# Task-specific data processors
def prepare_chat_dataset(examples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 1024):
    """Prepare dataset for chat fine-tuning"""
    processed_examples = []
    
    for example in examples:
        # Format conversation
        if "messages" in example:
            conversation = ""
            for message in example["messages"]:
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation += f"<|{role}|>{content}<|end|>\n"
            
            processed_examples.append({"text": conversation})
        elif "prompt" in example and "response" in example:
            text = f"<|user|>{example['prompt']}<|end|>\n<|assistant|>{example['response']}<|end|>"
            processed_examples.append({"text": text})
    
    return processed_examples


def prepare_instruct_dataset(examples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 1024):
    """Prepare dataset for instruction fine-tuning"""
    processed_examples = []
    
    for example in examples:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        processed_examples.append({"text": text})
    
    return processed_examples


def prepare_code_dataset(examples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048):
    """Prepare dataset for code fine-tuning"""
    processed_examples = []
    
    for example in examples:
        if "prompt" in example and "completion" in example:
            text = f"# Task: {example['prompt']}\n\n{example['completion']}"
        elif "problem" in example and "solution" in example:
            text = f"# Problem:\n{example['problem']}\n\n# Solution:\n{example['solution']}"
        elif "code" in example:
            text = example["code"]
        else:
            continue
        
        processed_examples.append({"text": text})
    
    return processed_examples 