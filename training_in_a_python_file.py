import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F
import time
from datetime import timedelta
# ------------------------------------------
# Base folder names
# ------------------------------------------
train_folder = "train_color_candlestick_vol_0_ma_1_bb_0_rsi_0"
test_folder  = "test_color_candlestick_vol_0_ma_1_bb_0_rsi_0"

# Choose how many tickers to use
# Set to None if you want all tickers
MAX_TICKERS = None

# Make ticker selection reproducible
TICKER_SEED = 42

BATCH_SIZE = 512
NUM_WORKERS = 4
PREFETCH_FACTOR = 2

PRINT_EVERY = 500

class StockImageDataset(Dataset):
    def __init__(self, df, transform=None):
        df = df.reset_index(drop=True)

        self.image_paths = df["image_path"].astype(str).to_numpy()
        self.labels = df["label"].astype("int64").to_numpy()

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as image:
            image = image.convert("RGB")  # convert PIL image to RGB
            if self.transform:
                image = self.transform(image)

        label = int(self.labels[idx])
        return image, label, idx

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Using padding='same' (requires recent PyTorch versions) to preserve spatial dimensions.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        self.dropout = nn.Dropout(p=0.5)
        # After two pooling layers:
        #   Input: 32x15
        #   After conv1 (same padding): 32x15, then pool1 (2x1): 16x15.
        #   After conv2 (same padding): 16x15, then pool2 (2x1): 8x15.
        # Flattened features: 128 * 8 * 15 = 15360.
        self.fc = nn.Linear(128 * 8 * 15, num_classes) # Jonas sendte dette men virker kun for 32x15
        # self.fc = nn.Linear(128 * 24 * 96, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, _) in enumerate(loader, 1):
        if batch_idx % PRINT_EVERY == 0 or batch_idx == 1 or batch_idx == len(loader):
            print(f"    Batch {batch_idx}/{len(loader)}")
            
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    script_start_time = time.perf_counter()
    # Build paths
    train_csv_path = os.path.join(train_folder, "labels.csv")
    test_csv_path  = os.path.join(test_folder, "labels.csv")

    # Load csv files
    train_full_df = pd.read_csv(train_csv_path)
    #train_full_df = train_full_df.iloc[4::5].copy() # hver 5 række bliver brugt, som starter ved 4 række, for ikke at få første række med
    test_full_df  = pd.read_csv(test_csv_path)

    # Force labels to be integers
    train_full_df["label"] = train_full_df["label"].astype(int)
    test_full_df["label"]  = test_full_df["label"].astype(int)

    # Convert dates
    for df_ in [train_full_df, test_full_df]:
        df_["start_date"] = pd.to_datetime(df_["start_date"])
        df_["end_date"]   = pd.to_datetime(df_["end_date"])
        df_["date"]       = df_["end_date"]   # use end_date as observation date

    # --------------------------------------------------
    # LIMIT NUMBER OF TICKERS
    # Pick tickers from the train set, then keep only
    # those same tickers in both train and test
    # --------------------------------------------------
    all_train_tickers = sorted(train_full_df["ticker"].unique())
    print(f"Total tickers available in train before filtering: {len(all_train_tickers)}")

    if MAX_TICKERS is not None:
        if MAX_TICKERS > len(all_train_tickers):
            raise ValueError(
                f"MAX_TICKERS={MAX_TICKERS} is larger than the number of available train tickers ({len(all_train_tickers)})."
            )

        rng = random.Random(TICKER_SEED)
        selected_tickers = sorted(rng.sample(all_train_tickers, MAX_TICKERS))

        train_full_df = train_full_df[train_full_df["ticker"].isin(selected_tickers)].copy()
        test_full_df  = test_full_df[test_full_df["ticker"].isin(selected_tickers)].copy()

        print(f"Using {len(selected_tickers)} tickers: {selected_tickers[:10]}{' ...' if len(selected_tickers) > 10 else ''}")

    print(f"\nRows after ticker filtering:")
    print(f"Train rows: {len(train_full_df)}")
    print(f"Test rows:  {len(test_full_df)}")

    print(f"\nUnique tickers after filtering:")
    print(f"Train tickers: {train_full_df['ticker'].nunique()}")
    print(f"Test tickers:  {test_full_df['ticker'].nunique()}")

    print("\nTrain head:")
    print(train_full_df.head())

    print("\nTrain info:")
    print(train_full_df.info())

    print("\nTrain label counts:")
    print(train_full_df["label"].value_counts())

    print("\nTest label counts:")
    print(test_full_df["label"].value_counts())

    # --------------------------------------------------
    # Split only the TRAIN folder into train/validation
    # Keep TEST folder as final out-of-sample test set
    # --------------------------------------------------

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=0.30,              # Jiang uses 70/30 train/val
        random_state=42,
        stratify=train_full_df["label"]
    )

    # The separate test folder is already the test set
    test_df = test_full_df.copy()

    # Optional: reset index
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print("Train shape:", train_df.shape)
    print("Val shape:  ", val_df.shape)
    print("Test shape: ", test_df.shape)

    print("\nDate ranges:")
    print("Train:", train_df["date"].min(), "to", train_df["date"].max())
    print("Val:  ", val_df["date"].min(), "to", val_df["date"].max())
    print("Test: ", test_df["date"].min(), "to", test_df["date"].max())

    print("\nLabel balance:")
    print(f"Train up: {train_df['label'].mean():.2%}")
    print(f"Val up:   {val_df['label'].mean():.2%}")
    print(f"Test up:  {test_df['label'].mean():.2%}")


    basic_transform = transforms.ToTensor()

    train_dataset = StockImageDataset(train_df, transform=basic_transform)
    val_dataset   = StockImageDataset(val_df, transform=basic_transform)
    test_dataset  = StockImageDataset(test_df, transform=basic_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR
    )

    # Test one batch
    images, labels, idx = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First 10 labels:", labels[:10])

    # Plot 3 resulting images
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for i in range(3):
        image, label, idx = train_dataset[i]
        img_np = image.numpy()

        if img_np.shape[0] == 1:
            axes[i].imshow(img_np[0], cmap="gray")
        else:
            axes[i].imshow(np.transpose(img_np, (1, 2, 0)))

        axes[i].set_title(f"Label: {label}, Index: {idx}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    print("Python:", sys.executable)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))



    

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SimpleCNN(
        #input_channels=INPUT_CHANNELS,
        #input_size=IMG_RESOLUTION,
        num_classes=2
    ).to(device)

    print("Trainable params:", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 50
    patience = 2   # Jiang-style early stopping after 2 non-improving epochs
    best_val_loss = float("inf")
    patience_counter = 0

    model_save_path = os.path.join(
        train_folder,
        f"best_jiang_baseline_i5.pth"
    )

    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.perf_counter() - epoch_start_time
        total_time = time.perf_counter() - script_start_time

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")
        print(f"  Epoch time: {timedelta(seconds=int(epoch_time))}")
        print(f"  Total time: {timedelta(seconds=int(total_time))}")
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("  Saved best model.")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break

    print("Best model path:", model_save_path)
    total_time = time.perf_counter() - script_start_time
    print(f"Finished training in: {timedelta(seconds=int(total_time))}")

if __name__ == "__main__":
    main()