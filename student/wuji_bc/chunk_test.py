from pathlib import Path

from .train import train
from .rollout import evaluate_policy

def train_policy():
    Path("videos/chunk_ablation").mkdir(parents=True, exist_ok=True)
    chunks = [1, 4, 8]
    print("Training the policies")
    for i in chunks:
        train(40000, 1000, 256, 0, i, wandb_project=f"chunk_ablation", wandb_run_name=f"chunk_{i}", ckpt_file=f"checkpoints/chunk_ablation/receding/chunk{i}.msgpack")


def main():
    #train_policy()
    chunks = [1,4,8]
    stats = {}
    print("Evaluating the policies")
    for i in chunks:
        stats[f'chunk{i}'] = []
        for ep in range(40):
            if ep < 19:
                stats[f'chunk{i}'].append(
                    evaluate_policy(1, ep, 10, checkpoint_file=f"checkpoints/chunk_ablation/receding/chunk{i}.msgpack")
                )
            else:
                stats[f'chunk{i}'].append(
                    evaluate_policy(1, ep, 10, video_path=f"videos/chunk_ablation/receding/chunk{i}.mp4", chunk_mode="receding", checkpoint_file=f"checkpoints/chunk_ablation/receding/chunk{i}.msgpack")
                )    
    print("Results (same seeds, 20 ep per chunk amount)")
    for i in chunks:
        successes = 0
        for ep in range(20):
            successes += stats[f'chunk{i}'][ep]['success_rate']
        print(f"Chunk {i}'s success rate: {successes/20}")

main()