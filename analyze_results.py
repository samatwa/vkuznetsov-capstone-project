import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Шлях згідно з main.py
file_path = "results/final_experiment_results.csv"

# Мікро-перевірка наявності файлу, якщо його немає, спробуємо знайти альтернативу
if not os.path.exists(file_path):
    if os.path.exists("final_experiment_results.csv"):
        file_path = "final_experiment_results.csv"
    else:
        print(f"Помилка: Файл {file_path} не знайдено.")
      
print(f"Читаємо дані з: {file_path}\n")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Не вдалося завантажити файл. Перевірте шляхи.")
    exit(1)


# 2. Функція генерації зведеної таблиці
def generate_summary(dataset_name):
    # Фільтруємо дані
    ds_df = df[df["dataset"] == dataset_name]

    if ds_df.empty:
        print(f"Дані для {dataset_name} не знайдено у файлі.")
        return pd.DataFrame()

    # Беремо дані лише за останню (30-ту) епоху для точності тестових та валідаційних метрик
    # Знаходимо останній запис для кожної активації (це має бути фінальна епоха)
    last_epoch = ds_df.loc[ds_df.groupby("activation")["epoch"].idxmax()]

    # Рахуємо середні значення для часу та стабільності градієнтів за всі епохи
    # Також беремо максимальну пікову пам'ять
    averages = (
        ds_df.groupby("activation")
        .agg({"epoch_time": "mean", "avg_grad_norm": "mean", "peak_memory_mb": "max"})
        .reset_index()
    )

    # Об'єднуємо дані: беремо test_acc/val_acc з останньої епохи, а решту - агреговані
    summary = pd.merge(
        last_epoch[["activation", "test_acc", "val_acc"]], averages, on="activation"
    )

    # Сортуємо за тестовою точністю за спаданням та округлюємо
    summary = summary.sort_values(by="test_acc", ascending=False)
    return summary.round(4)


# 3. Генерація таблиць
cifar_summary = generate_summary("CIFAR10")
agnews_summary = generate_summary("AGNews")

if not cifar_summary.empty:
    print("\n=== Таблиця 4.1: Результати для ResNet-34 (CIFAR-10) ===")
    print(cifar_summary.to_markdown(index=False))

if not agnews_summary.empty:
    print("\n=== Таблиця 4.2: Результати для Transformer Encoder (AG News) ===")
    print(agnews_summary.to_markdown(index=False))


# 4. ВІЗУАЛІЗАЦІЯ (Trade-off: Точність проти Часу)
def plot_tradeoff(summary_df, title):
    if summary_df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=summary_df,
        x="epoch_time",
        y="test_acc",
        hue="activation",
        s=300,
        palette="Set1",
        legend=False,  # Вимикаємо легенду тут, бо підпишемо точки вручну
    )

    # Підписи точок
    for i in range(summary_df.shape[0]):
        plt.text(
            summary_df["epoch_time"].iloc[i],
            summary_df["test_acc"].iloc[i],
            f" {summary_df['activation'].iloc[i].capitalize()}",
            horizontalalignment="left",
            verticalalignment="center",
            size="medium",
            color="black",
            weight="bold",
        )

    plt.title(f"Trade-off: Точність vs Час на епоху ({title})", fontsize=14, pad=15)
    plt.xlabel("Середній час на епоху (секунди)", fontsize=12)
    plt.ylabel("Тестова точність (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Збереження графіка
    filename = f"results/tradeoff_{title.replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"\nГрафік збережено: {filename}")
    plt.close()


# Будуємо графіки
if not cifar_summary.empty:
    plot_tradeoff(cifar_summary, "ResNet-34 / CIFAR-10")

if not agnews_summary.empty:
    plot_tradeoff(agnews_summary, "Transformer / AG News")
