import torch
import matplotlib.pyplot as plt
import time
import argparse
from cs336_basics import adapters
from cs336_basics import transformer_lm
from cs336_basics.train import build_model, build_optimizer, cross_entropy, build_optimizer

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
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
    
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)

    return parser.parse_args()

import statistics
import matplotlib.pyplot as plt

def benchmarking(args: argparse.Namespace):
    batch_size = 4
    warmup_steps = 5
    forward_steps = 10
    backward_steps = 10
    optim_steps = 10
    model = build_model(args)
    optimizer = build_optimizer(model, args)
    loss_fn = cross_entropy
    data = torch.randint(0, args.vocab_size,
                         (batch_size, args.context_length + 1),
                         device=args.device)
    inputs = data[:, :-1]
    targets = data[:, 1:]

    forward_times = []
    backward_times = []
    optim_times = []

    ## forward
    with torch.no_grad():
        for _ in range(warmup_steps):
            model(inputs)
        torch.cuda.synchronize()

        for _ in range(forward_steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            forward_times.append((end - start) * 1000)

    ## forward + loss + backward
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, args.vocab_size),
                       targets.reshape(-1))
        loss.backward()
        del logits, loss
    torch.cuda.synchronize()

    for _ in range(backward_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, args.vocab_size),
                       targets.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        backward_times.append((end - start) * 1000)
        del logits, loss

    ## full training step
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, args.vocab_size),
                       targets.reshape(-1))
        loss.backward()
        optimizer.step()
        del logits, loss
    torch.cuda.synchronize()

    for _ in range(optim_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, args.vocab_size),
                       targets.reshape(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        optim_times.append((end - start) * 1000)
        del logits, loss

    ## statistics and plot
    results = [
        ("Forward (no grad)", forward_times),
        ("Forward + loss + backward", backward_times),
        ("Full training step", optim_times),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (label, timings) in zip(axes, results):
        mean = statistics.mean(timings)
        std = statistics.stdev(timings)
        print(f"{label}: {mean:.3f} ± {std:.3f} ms (sample std)")
        print("  Steps (ms):", ", ".join(f"{t:.3f}" for t in timings))

        steps = range(1, len(timings) + 1)
        ax.plot(steps, timings, marker="o", label="Step time")
        ax.axhline(mean, color="tab:orange", linestyle="--",
                   label="Mean")
        ax.axhspan(mean - std, mean + std,
                   color="tab:orange", alpha=0.2, label="Mean ± std")
        ax.set_title(f"{label}\n{mean:.3f} ± {std:.3f} ms")
        ax.set_xlabel("Measurement step")
        ax.set_xticks(list(steps))
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Batch={batch_size}, context={args.context_length}, "
        f"d_model={args.d_model}, layers={args.num_layers}, "
        f"heads={args.num_heads}, d_ff={args.d_ff}, "
        f"vocab={args.vocab_size}\n"
        f"Warm-up={warmup_steps}; measurements per stage=10"
    )
    fig.tight_layout()
    fig.savefig("./figures/small_benchmark_times.png", dpi=200)
    plt.close(fig)
    print("Saved plot to ./figures/small_benchmark_times.png")
    
    
    
if __name__ == "__main__":
    args = parse_arg()
    benchmarking(args)