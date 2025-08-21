import os, torch, json
import random
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
from huggingface_hub import hf_hub_download

def load_data(filename: str) -> List[Dict[str, Any]]:
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def create_dataloader(filename: str, batch_size: int, shuffle: bool = True, sort_by_length: bool = False) -> DataLoader:
    data = load_data(filename)
    
    # Load tokenizer for character to token mapping
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    
    examples = []
    for d in data:
        response_type = d["response_type"]
        assistant_response = d["assistant_response"]
        exact_answer = d.get("exact_answer", None)
        
        # Default positions (for non-answer attempts or when answer not found)
        start_pos, end_pos = 0, 0
        
        if exact_answer is not None and response_type == "answer_attempt":
            # Find the exact_answer in the assistant_response
            if exact_answer in assistant_response:
                # Get character positions
                char_start_pos = assistant_response.find(exact_answer)
                char_end_pos = char_start_pos + len(exact_answer) - 1
                
                # Tokenize the response
                tokenized = tokenizer(assistant_response, return_offsets_mapping=True, add_special_tokens=True)
                offsets = tokenized["offset_mapping"]
                
                # Map character positions to token positions
                # Special tokens (like CLS) have offset (0, 0)
                # First find the tokens that contain the start and end positions
                start_pos, end_pos = 0, 0
                for idx, (token_start, token_end) in enumerate(offsets):
                    # Skip special tokens which have offset (0, 0)
                    if token_start == 0 and token_end == 0:
                        continue
                    
                    # Token contains the start position
                    if token_start <= char_start_pos < token_end:
                        start_pos = idx
                    
                    # Token contains the end position
                    if token_start <= char_end_pos < token_end:
                        end_pos = idx
                        break
            
        examples.append({
            "assistant_response": assistant_response,
            "response_type": response_type,
            "exact_answer": exact_answer,
            "start_pos": start_pos,
            "end_pos": end_pos
        })
    
    # Sort examples by length of assistant_response if requested
    if sort_by_length:
        examples = sorted(examples, key=lambda x: len(x['assistant_response']))
        shuffle = False  # Override shuffle when sorting
    
    def collate_fn(batch):
        assistant_responses = [x['assistant_response'] for x in batch]
        response_types = [x['response_type'] for x in batch]
        start_positions = torch.LongTensor([x['start_pos'] for x in batch])
        end_positions = torch.LongTensor([x['end_pos'] for x in batch])
        
        return assistant_responses, response_types, start_positions, end_positions
    
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

class MBertConvStrat(nn.Module):
    def __init__(self, model_name: str, max_length: int = 4096):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # Load tokenizer and NLU model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlu = AutoModel.from_pretrained(model_name)
        hidden_size = self.nlu.config.hidden_size
        
        # Response types
        self.response_types = ["answer_attempt", "discussion", "refuse", "clarification", 
                               "interrogation", "hedge", "missing"]
        self.response_type_to_idx = {rt: i for i, rt in enumerate(self.response_types)}
        
        # Response type classification head (7-way classification)
        self.response_type_head = self._create_response_type_head(hidden_size)
        
        # Span extraction head for answer extraction
        self.span_start_head = nn.Linear(hidden_size, 1)
        self.span_end_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights(self.response_type_head)
        self._init_weights(self.span_start_head)
        self._init_weights(self.span_end_head)
        
        # Try to load the custom heads from the model
        self._load_custom_heads(model_name)
        
        self.to(self.device)

    def _find_token_positions(self, text: str, answer: str) -> Tuple[int, int]:
        """Find token positions for an answer in a text."""
        if answer not in text:
            return 0, 0
            
        # Find character positions
        char_start_pos = text.find(answer)
        char_end_pos = char_start_pos + len(answer) - 1
        
        # Tokenize the response
        tokenized = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        print(tokenized)
        offsets = tokenized["offset_mapping"]
        
        # Map character positions to token positions
        start_pos, end_pos = 0, 0
        for idx, (token_start, token_end) in enumerate(offsets):
            # Skip special tokens which have offset (0, 0)
            if token_start == 0 and token_end == 0:
                continue
            
            # Token contains the start position
            if token_start <= char_start_pos < token_end:
                start_pos = idx
            
            # Token contains the end position
            if token_start <= char_end_pos < token_end:
                end_pos = idx
                break
        
        return start_pos, end_pos

    def _create_response_type_head(self, hidden_size):
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, len(self.response_types)),
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _load_custom_heads(self, model_name: str):
        """Load custom heads from local or online models"""
        # Check if model_name is a local path or an online model
        if os.path.exists(model_name):
            # Local path
            heads_path = os.path.join(model_name, 'heads.pth')
            if os.path.exists(heads_path):
                heads_state = torch.load(heads_path, map_location=self.device)
                self.response_type_head.load_state_dict(heads_state['response_type_head'])
                self.span_start_head.load_state_dict(heads_state['span_start_head'])
                self.span_end_head.load_state_dict(heads_state['span_end_head'])

    
    def forward(self, assistant_responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokenize all responses
        encoded = self.tokenizer(assistant_responses, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)

        # Process through model
        outputs = self.nlu(**encoded)
        
        # Get sequence output for classification
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # Use CLS token for classification
        
        # Response type classification
        response_type_logits = self.response_type_head(pooled_output)
        
        # Span extraction
        start_logits = self.span_start_head(sequence_output).squeeze(-1)
        end_logits = self.span_end_head(sequence_output).squeeze(-1)
        
        return response_type_logits, start_logits, end_logits, encoded.input_ids, encoded.attention_mask
    
    def predict(self, assistant_response: str) -> Dict[str, Any]:
        """Predict response type and extract answer if applicable"""
        self.eval()
        with torch.no_grad():
            response_type_logits, start_logits, end_logits, input_ids, attention_mask = self([assistant_response])
            
            # Get response type prediction
            response_type_probs = torch.softmax(response_type_logits, dim=1)[0]
            response_type_idx = torch.argmax(response_type_probs, dim=0).item()
            response_type = self.response_types[response_type_idx]
            
            # Create probability dictionary for each class
            class_probabilities = {rt: response_type_probs[i].item() for i, rt in enumerate(self.response_types)}
            
            result = {
                "response_type": response_type,
                "class_probabilities": class_probabilities
            }
            
            # Extract answer if it's an answer attempt
            if response_type == "answer_attempt":
                # Get span predictions (masked by attention)
                masked_start_logits = start_logits[0] * attention_mask[0]
                masked_end_logits = end_logits[0] * attention_mask[0]
                
                start_idx = torch.argmax(masked_start_logits[1:]).item() + 1 # force to avoid the first token
                end_idx = torch.argmax(masked_end_logits[1:]).item() + 1 # force to avoid the first token
                
                # Ensure end comes after start
                if end_idx < start_idx:
                    end_idx = start_idx

                # Get token IDs and convert to text
                answer_token_ids = input_ids[0][start_idx:end_idx+1]
                answer_text = self.tokenizer.decode(answer_token_ids, skip_special_tokens=True)
                
                result["exact_answer"] = answer_text
                result["start_pos"] = start_idx
                result["end_pos"] = end_idx
            
            return result
    
    def evaluate(self, val_loader):
        self.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        span_mse = 0.0
        answer_attempt_count = 0
        
        response_type_metrics = {rt: {"tp": 0, "fp": 0, "fn": 0} for rt in self.response_types}
        
        with torch.no_grad():
            for assistant_responses, response_types, start_positions, end_positions in val_loader:
                response_type_logits, start_logits, end_logits, input_ids, attention_mask = self(assistant_responses)
                
                # Convert string labels to indices
                label_indices = torch.tensor([self.response_type_to_idx[rt] for rt in response_types], 
                                             device=self.device)
                
                # Calculate classification loss
                cls_loss = nn.CrossEntropyLoss()(response_type_logits, label_indices)
                
                # Track classification accuracy
                predictions = torch.argmax(response_type_logits, dim=1)
                correct_predictions += (predictions == label_indices).sum().item()
                total_examples += len(assistant_responses)
                
                # Calculate span extraction loss for answer attempts
                span_loss = 0.0
                for i, (pred, true_type, start_pos, end_pos) in enumerate(
                    zip(predictions, response_types, start_positions, end_positions)):
                    
                    pred_type = self.response_types[pred.item()]
                    
                    # Update metrics for each response type
                    for rt in self.response_types:
                        if true_type == rt and pred_type == rt:
                            response_type_metrics[rt]["tp"] += 1
                        elif pred_type == rt and true_type != rt:
                            response_type_metrics[rt]["fp"] += 1
                        elif true_type == rt and pred_type != rt:
                            response_type_metrics[rt]["fn"] += 1
                    
                    if true_type == "answer_attempt":
                        answer_attempt_count += 1
                        # Convert positions to device
                        start_pos = start_pos.to(self.device)
                        end_pos = end_pos.to(self.device)
                        
                        # For answer attempts, calculate span loss
                        start_loss = nn.CrossEntropyLoss()(start_logits[i].unsqueeze(0), 
                                                           start_pos.unsqueeze(0))
                        end_loss = nn.CrossEntropyLoss()(end_logits[i].unsqueeze(0), 
                                                         end_pos.unsqueeze(0))
                        
                        this_span_loss = (start_loss + end_loss) / 2
                        span_loss += this_span_loss
                        
                        # Track span prediction error
                        pred_start = torch.argmax(start_logits[i] * attention_mask[i]).item()
                        pred_end = torch.argmax(end_logits[i] * attention_mask[i]).item()
                        
                        span_mse += ((pred_start - start_pos.item())**2 + 
                                     (pred_end - end_pos.item())**2) / 2
                
                # Only include span loss if there are answer attempts
                if answer_attempt_count > 0:
                    span_loss = span_loss / answer_attempt_count
                    total_loss += cls_loss + span_loss
                else:
                    total_loss += cls_loss
        
        # Calculate metrics
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_span_mse = span_mse / answer_attempt_count if answer_attempt_count > 0 else 0
        
        # Calculate F1 for each response type
        f1_scores = {}
        for rt in self.response_types:
            metrics = response_type_metrics[rt]
            precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if metrics["tp"] + metrics["fp"] > 0 else 0
            recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if metrics["tp"] + metrics["fn"] > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            f1_scores[rt] = f1
        
        avg_f1 = sum(f1_scores.values()) / len(f1_scores)
        
        return {
            "loss": avg_loss.item(),
            "accuracy": accuracy,
            "span_mse": avg_span_mse,
            "f1_scores": f1_scores,
            "avg_f1": avg_f1,
            "answer_attempt_count": answer_attempt_count,
            "total_examples": total_examples
        }
    
    def save_model(self, save_dir):
        """Save the model to a directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the base model and tokenizer
        self.nlu.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save the custom heads
        heads_state = {
            'response_type_head': self.response_type_head.state_dict(),
            'span_start_head': self.span_start_head.state_dict(),
            'span_end_head': self.span_end_head.state_dict()
        }
        torch.save(heads_state, os.path.join(save_dir, 'heads.pth'))
    
    @classmethod
    def load_model(cls, model_path):
        model = cls(model_path)
        return model

if __name__ == "__main__":
    # Example usage
    model_name = "models/mbert-conv-strat-loss0.611/"
    model = MBertConvStrat(model_name)

    import json, random

    with open("data/conv_strategy_validation.json", "r") as f:
        data = json.load(f)

    responses = [d["assistant_response"] for d in data if d["response_type"] == "answer_attempt"]

    random.shuffle(responses)

    responses = responses[:10]

    for response in responses:
        print("=============================")
        prediction = model.predict(response)

        response_text = response
        if "exact_answer" in prediction:
            # make it in blue
            response_text = response_text.replace(prediction["exact_answer"], f"\033[94m{prediction['exact_answer']}\033[0m")

        print(response_text)

        print('----')
        for k in sorted(prediction["class_probabilities"].items(), key=lambda x: x[1], reverse=True):
            print(f"   {k[0]}: {k[1]:.2f}")
        
    # Example of token mapping
    # answer = "Paris"
    # start_token, end_token = model._find_token_positions(response, answer)
    # print(f"Answer: '{answer}' found at token positions {start_token} to {end_token}") 