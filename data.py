import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from datasets import load_dataset
from collections import Counter

# Фіксація Random Seed
torch.manual_seed(42)
np.random.seed(42)


def get_cifar10_loaders(batch_size=128, val_split=0.1, num_workers=0):
    """
    Повертає завантажувачі даних CIFAR-10: train, val, test.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    np.random.shuffle(indices)

    val_idx, train_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_ag_news_loaders(batch_size=64, max_len=256, vocab_size=20000, num_workers=0):
    """
    Повертає завантажувачі даних AG News за допомогою бібліотеки HuggingFace datasets.
    Повертає: train_loader, val_loader, test_loader, vocab_len

    ПРИМІТКА: collate_fn повертає тензори у форматі [batch, seq_len] (batch_first=True)
    щоб відповідати оновленій TransformerModel з batch_first=True.
    """
    print("Loading AG News via HuggingFace...")
    dataset = load_dataset("ag_news")
    full_train_data = dataset["train"]
    test_data = dataset["test"]

    def tokenizer(text):
        return str(text).lower().split()

    counter = Counter()
    for item in full_train_data:
        counter.update(tokenizer(item["text"]))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    def text_pipeline(text):
        return [vocab.get(token, vocab["<unk>"]) for token in tokenizer(text)]

    def collate_batch(batch):
        label_list, text_list = [], []
        for item in batch:
            label_list.append(item["label"])
            processed_text = torch.tensor(
                text_pipeline(item["text"]), dtype=torch.int64
            )
            if processed_text.size(0) > max_len:
                processed_text = processed_text[:max_len]
            text_list.append(processed_text)

        label_list = torch.tensor(label_list, dtype=torch.int64)

        # batch_first=True: вихідна форма [batch, seq_len]
        # Це відповідає TransformerModel, яка тепер очікує [batch, seq_len]
        text_list = pad_sequence(
            text_list, batch_first=True, padding_value=vocab["<pad>"]
        )
        return text_list, label_list

    # Розділення повного тренувального набору на train/val (90/10)
    num_train = len(full_train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.shuffle(indices)

    val_idx, train_idx = indices[:split], indices[split:]

    train_subset = Subset(full_train_data, train_idx)
    val_subset = Subset(full_train_data, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, len(vocab)
