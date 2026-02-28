import pandas as pd
import torch
import numpy as np
import random
from models.resnet import ResNet34
from models.transformer import TransformerModel
from data import get_cifar10_loaders, get_ag_news_loaders
from train_utils import run_experiment
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)

    activations = [
        "relu",
        "leaky_relu",
        "elu",
        "selu",
        "gelu",
        "swish",
        "mish",
        "hardswish",
        "softplus",
    ]

    experiments = [
        {
            "dataset_name": "CIFAR10",
            "model_fn": ResNet34,
            "dataset_fn": get_cifar10_loaders,
            "batch_size": 128,
            "learning_rate": 0.001,
        },
        {
            "dataset_name": "AGNews",
            "model_fn": TransformerModel,
            "dataset_fn": get_ag_news_loaders,
            "batch_size": 64,
            "learning_rate": 0.0001,
        },
    ]

    output_dir = "results"
    checkpoints_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    all_results = []

    for exp_config in experiments:
        dataset_name = exp_config["dataset_name"]
        print(f"\n{'='*60}")
        print(f"Starting experiments for {dataset_name}")
        print(f"{'='*60}")

        for activation in activations:
            print(f"\n--- Activation: {activation} ---")

            try:
                exp_results, model_state = run_experiment(
                    model_fn=exp_config["model_fn"],
                    dataset_fn=exp_config["dataset_fn"],
                    dataset_name=dataset_name,
                    activation=activation,
                    epochs=2,  # Повний цикл із 30 епох для фінальних результатів
                    batch_size=exp_config["batch_size"],
                    learning_rate=exp_config["learning_rate"],
                )

                all_results.extend(exp_results)

                # Збереження контрольної точки
                checkpoint_path = os.path.join(
                    checkpoints_dir, f"checkpoint_{dataset_name}_{activation}.pth"
                )
                torch.save(model_state, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

                # Збереження проміжних результатів після кожної активації
                df = pd.DataFrame(all_results)
                df.to_csv(f"{output_dir}/results_intermediate.csv", index=False)

                # Збереження CSV для кожної активації
                act_df = pd.DataFrame(exp_results)
                act_df.to_csv(
                    f"{output_dir}/results_{dataset_name}_{activation}.csv", index=False
                )

            except Exception as e:
                print(f"Experiment failed for {dataset_name} - {activation}: {e}")
                import traceback

                traceback.print_exc()

    # Фінальне збереження
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(f"{output_dir}/final_experiment_results.csv", index=False)
    print("\nAll experiments completed.")
    print(f"Results saved to {output_dir}/final_experiment_results.csv")


if __name__ == "__main__":
    main()
