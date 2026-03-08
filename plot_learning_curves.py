import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_learning_curves(df, dataset_name, metric="val_acc"):
    """
    Будує графіки динаміки навчання (Learning Curves) по епохах.
    metric: 'val_acc', 'val_loss', 'train_acc', 'train_loss'
    """
    data = df[df["dataset"] == dataset_name]

    if data.empty:
        print(f"Дані для {dataset_name} відсутні.")
        return

    plt.figure(figsize=(12, 8))

    # Побудова лінійного графіка
    sns.lineplot(
        data=data,
        x="epoch",
        y=metric,
        hue="activation",
        style="activation",  # Додає різні стилі ліній/маркерів для кращої розрізнюваності
        markers=True,
        dashes=False,
        linewidth=2,
        palette="tab10",
    )

    # Налаштування підписів українською
    metric_labels = {
        "val_acc": "Точність на валідації (%)",
        "train_acc": "Точність на тренуванні (%)",
        "val_loss": "Втрати на валідації",
        "train_loss": "Втрати на тренуванні",
    }

    metric_title_part = {
        "val_acc": "Validation Accuracy",
        "train_acc": "Training Accuracy",
        "val_loss": "Validation Loss",
        "train_loss": "Training Loss",
    }

    ylabel = metric_labels.get(metric, metric)
    title_metric = metric_title_part.get(metric, metric)

    model_name = (
        "ResNet-34"
        if dataset_name == "CIFAR10"
        else "Transformer" if dataset_name == "AGNews" else "Model"
    )

    plt.title(
        f"Динаміка навчання: {title_metric}\n({model_name} / {dataset_name})",
        fontsize=16,
    )
    plt.xlabel("Епоха", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Функція активації", fontsize=12, title_fontsize=12, loc="best")

    # Встановлення цілих чисел для осі X (епохи)
    max_epoch = data["epoch"].max()
    plt.xticks(
        range(1, max_epoch + 1, 2 if max_epoch > 20 else 1)
    )  # Крок 2, якщо епох багато

    plt.tight_layout()

    # Збереження
    os.makedirs("results", exist_ok=True)
    filename = f"results/learning_curve_{dataset_name}_{metric}.png"
    plt.savefig(filename, dpi=300)
    print(f"Графік збережено: {filename}")
    plt.show()
    plt.close()


def main():
    file_path = "results/final_experiment_results.csv"

    # Перевірка шляхів (як і в інших скриптах)
    if not os.path.exists(file_path):
        if os.path.exists("final_experiment_results.csv"):
            file_path = "final_experiment_results.csv"
        else:
            print("Файл з результатами не знайдено.")
            return

    print(f"Завантаження даних з {file_path}...")
    df = pd.read_csv(file_path)

    datasets = df["dataset"].unique()

    for ds in datasets:
        # 1. Динаміка точності на валідації (найважливіший графік)
        plot_learning_curves(df, ds, "val_acc")

        # 2. Динаміка втрат на валідації (для аналізу збіжності/перенавчання)
        plot_learning_curves(df, ds, "val_loss")

        # Можна додати train_acc/train_loss за бажанням, але перші два найбільш ілюстративні


if __name__ == "__main__":
    main()
