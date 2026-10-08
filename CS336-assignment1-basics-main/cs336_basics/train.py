import torch
import torch.nn as nn
from dataclasses import dataclass
import argparse
import numpy as np


from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import Transformer_lm
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.cosine_annealing import cosine_annealing
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.data_loading import data_loading
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 1600
    num_layers: int = 48
    num_heads: int = 25
    d_ff: int = 4288
    rope_theta: float = 10000.0

@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1.0
    grad_clip: float = 1.0
    num_steps: int = 100_000

def parse_args():
    parser = argparse.ArgumentParser()
    
    ## Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    ## Molde
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--d_ff", type=int, default=4288)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    ## Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    ## Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    ## Device
    parser.add_argument("--device", type=str, default="cpu")
    ## Log
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()

def load_data(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

def build_model(args: argparse.Namespace):
    return Transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta
    ).to(args.device)
    
def build_optimizer(model: nn.Module, args: argparse.Namespace):
    return AdamW(
        model.parameters(),
        lr = args.lr,
        betas = tuple(args.betas),
        eps = args.eps,
        weight_decay = args.weight_decay
    )
@torch.no_grad()    
def evaluate(model: nn.Module, val_data: np.ndarray, args: argparse.Namespace, num_batches: int = 20):
    model.eval()
    loss_fn = cross_entropy
    total_loss = 0.0
    for _ in range(num_batches):
        inputs, targets = data_loading(
            val_data, args.batch_size, args.context_length, args.device
        )
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))
        total_loss += loss.item()
    return total_loss / num_batches

def train(args: argparse.Namespace):
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    model = build_model(args)
    optimizer = build_optimizer(model, args)
    
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from step {start_step}")
    if args.wandb:
        import wandb
        wandb.init(project="cs336", config=vars(args))
    
    loss_fn = cross_entropy
    
    model.train()
    
    for step in range(start_step, args.num_steps):
        
        lr = cosine_annealing(
            step,
            args.lr_max,
            args.lr_min,
            args.warmup_steps,
            args.num_steps,
        )
        
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = data_loading(
            train_data, args.batch_size, args.context_length, args.device
        )
        
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        gradient_clipping(list(model.parameters()), args.grad_clip)
        
        optimizer.step()
        
        if step % args.log_interval == 0:
            print(f"step {step} | loss {loss.item():.4f} | lr {lr:.2e}")
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step = step)
                
        if step % args.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_data, args)
            print(f"step {step} | val loss {val_loss: .4f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss}, step)
            model.train()
            
        if step % args.save_interval ==0 and step > 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Saved checkpoint at step {step}")
    
if __name__ == "__main__":
    args = parse_args()
    train(args)