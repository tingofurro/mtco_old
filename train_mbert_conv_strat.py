from model_mbert_conv_strat import MBertConvStrat, create_dataloader
from transformers import get_linear_schedule_with_warmup
import argparse, os, shutil, torch, wandb, tqdm
from torch.optim import AdamW

def main():
    parser = argparse.ArgumentParser(description='Train ModernBERT conversation strategy classifier')
    parser.add_argument('--model', type=str, default="answerdotai/ModernBERT-large")
    parser.add_argument('--train_fn', type=str, default="data/conv_strategy_train.json")
    parser.add_argument('--val_fn', type=str, default="data/conv_strategy_validation.json")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_every', type=int, default=100)
    
    args = parser.parse_args()

    args.optim_every = 1
    if args.batch_size > 8:
        args.optim_every = args.batch_size // 8
        args.batch_size = (args.batch_size // args.optim_every)
        print(f'Batch size set to {args.batch_size} and optim_every set to {args.optim_every}')

    wandb.init(
        project="mtco-mbert-cls",
        config={"train_fn": args.train_fn, "val_fn": args.val_fn, "learning_rate": args.learning_rate, 
                "max_grad_norm": args.max_grad_norm, "epochs": args.epochs, "batch_size": args.batch_size, 
                "eval_every": args.eval_every, "model": args.model, "optim_every": args.optim_every}
        )
    wandb.run.name = f'mbert-conv-strat-lr{args.learning_rate:.1e}-bs{args.batch_size}-oe{args.optim_every}'

    model = MBertConvStrat(args.model)
    train_loader = create_dataloader(args.train_fn, args.batch_size)
    val_loader = create_dataloader(args.val_fn, 64, shuffle=False, sort_by_length=True)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) // 10,  # Add warmup steps (10% of first epoch)
        num_training_steps=len(train_loader) * args.epochs
    )

    best_loss = 100.0
    for epoch in range(args.epochs):
        for batch_idx, (assistant_responses, response_types, start_positions, end_positions) in enumerate(tqdm.tqdm(train_loader)):
            # Forward pass
            response_type_logits, start_logits, end_logits, input_ids, attention_mask = model(assistant_responses)
            
            # Convert string labels to indices
            label_indices = torch.tensor([model.response_type_to_idx[rt] for rt in response_types], 
                                         device=model.device)
            
            # Classification loss
            cls_loss = torch.nn.CrossEntropyLoss()(response_type_logits, label_indices)
            
            # Span extraction loss (only for answer attempts)
            span_loss = 0.0
            answer_attempt_count = 0
            
            for i, (response_type, start_pos, end_pos) in enumerate(zip(response_types, start_positions, end_positions)):
                if response_type == "answer_attempt":
                    answer_attempt_count += 1
                    # Convert positions to device
                    start_pos = start_pos.to(model.device)
                    end_pos = end_pos.to(model.device)
                    
                    # For answer attempts, calculate span loss
                    start_loss = torch.nn.CrossEntropyLoss()(start_logits[i].unsqueeze(0), 
                                                           start_pos.unsqueeze(0))
                    end_loss = torch.nn.CrossEntropyLoss()(end_logits[i].unsqueeze(0), 
                                                         end_pos.unsqueeze(0))
                    
                    this_span_loss = (start_loss + end_loss) / 2
                    span_loss += this_span_loss
                else:
                    # For non-answer attempts, train span heads to predict position 0
                    # print(response_type, start_logits[i].unsqueeze(0).shape)
                    start_pos = torch.tensor([0], device=model.device)
                    end_pos = torch.tensor([0], device=model.device)
                    
                    start_loss = torch.nn.CrossEntropyLoss()(start_logits[i].unsqueeze(0), 
                                                           start_pos)
                    end_loss = torch.nn.CrossEntropyLoss()(end_logits[i].unsqueeze(0), 
                                                         end_pos)
                    
                    this_span_loss = (start_loss + end_loss) / 2
                    span_loss += this_span_loss
            
            # Average span loss across batch
            span_loss = span_loss / len(response_types)
            
            # Combined loss
            loss = cls_loss + span_loss
            
            # Scale loss for gradient accumulation
            loss = loss / args.optim_every
            
            # Log metrics
            train_log = {
                'train/loss': loss.item(), 
                'train/cls_loss': cls_loss.item(),
                'train/span_loss': span_loss.item(),
                'train/answer_attempt_count': answer_attempt_count,
                'train/batch_size': len(response_types)
            }
            wandb.log(train_log)
            
            # Backward pass
            loss.backward()

            # Optimize every optim_every steps
            if (batch_idx + 1) % args.optim_every == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Evaluate
            if (batch_idx + 1) % args.eval_every == 0:
                val_metrics = model.evaluate(val_loader)
                
                wandb.log({
                    'val/loss': val_metrics["loss"],
                    'val/accuracy': val_metrics["accuracy"],
                    'val/span_mse': val_metrics["span_mse"],
                    'val/avg_f1': val_metrics["avg_f1"],
                    'val/answer_attempt_count': val_metrics["answer_attempt_count"],
                    'val/total_examples': val_metrics["total_examples"],
                })
                
                # Log F1 scores for each response type
                for rt, f1 in val_metrics["f1_scores"].items():
                    wandb.log({f'val/f1_{rt}': f1})
                
                print(f'Epoch {epoch+1}/{args.epochs}, Val Loss: {val_metrics["loss"]:.4f}, '
                      f'Accuracy: {val_metrics["accuracy"]:.4f}, Avg F1: {val_metrics["avg_f1"]:.4f}, '
                      f'Span MSE: {val_metrics["span_mse"]:.4f}')
                
                # Save model if validation loss improves
                if val_metrics["loss"] < best_loss:
                    # Delete previous best model folder if it exists
                    if hasattr(model, 'best_model_dir') and os.path.exists(model.best_model_dir):
                        shutil.rmtree(model.best_model_dir)
                    
                    best_loss = val_metrics["loss"]
                    save_dir = f'models/mbert-conv-strat-loss{best_loss:.3f}'
                    print(f'\033[94mSaving model to {save_dir}\033[0m')
                    model.best_model_dir = save_dir
                    model.save_model(save_dir)
                
                model.train()

    wandb.finish()

if __name__ == '__main__':
    main() 