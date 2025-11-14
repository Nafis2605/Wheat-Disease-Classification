# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# %%
# Load the data
train_images = np.load('../dataset/train_images.npy')
train_labels = np.load('../dataset/train_labels.npy')
val_images = np.load('../dataset/val_images.npy')
val_labels = np.load('../dataset/val_labels.npy')
test_images = np.load('../dataset/test_images.npy')
test_labels = np.load('../dataset/test_labels.npy')

# %%
# Normalize and preprocess the images
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def filter_classes(images, labels, exclude_classes):
    """
    Filters out specific classes from the dataset.

    Args:
        images (numpy.ndarray): Dataset images.
        labels (numpy.ndarray): Corresponding labels.
        exclude_classes (list): Classes to exclude.

    Returns:
        filtered_images: Images after excluding classes.
        filtered_labels: Labels after excluding classes.
    """
    mask = ~np.isin(labels, exclude_classes)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    return filtered_images, filtered_labels


def reindex_classes(labels):
    """
    Reindexes class labels to ensure they are consecutive after filtering.

    Args:
        labels (numpy.ndarray): Class labels.

    Returns:
        reindexed_labels: Labels with consecutive indexing.
        class_mapping: Mapping from old to new labels.
    """
    unique_classes = np.unique(labels)
    class_mapping = {old: new for new, old in enumerate(unique_classes)}
    reindexed_labels = np.vectorize(class_mapping.get)(labels)
    return reindexed_labels, class_mapping

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean/std for grayscale or RGB as needed
])

# exclude_classes = []  # Replace with the classes you want to exclude
# train_images, train_labels = filter_classes(train_images, train_labels, exclude_classes)
# val_images, val_labels = filter_classes(val_images, val_labels, exclude_classes)
# test_images, test_labels = filter_classes(test_images, test_labels, exclude_classes)

# train_labels, class_mapping = reindex_classes(train_labels)
# val_labels, _ = reindex_classes(val_labels)
# test_labels, _ = reindex_classes(test_labels)
# print(f"Class Mapping: {class_mapping}")

# Create datasets and dataloaders
train_dataset = CustomDataset(train_images, train_labels, transform=transform)
val_dataset = CustomDataset(val_images, val_labels, transform=transform)
test_dataset = CustomDataset(test_images, test_labels, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print(f"Input shape: {train_images.shape}")

# %%
# Define the model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 256 channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels=3,  # Match input channel count
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class EarlyStopping:
    def __init__(self, patience=30, delta=0, path="best_model.pth", verbose=False, val_loss_min=float("inf")):
        """
        Args:
            patience (int): How long to wait after last improvement.
            delta (float): Minimum change to qualify as improvement.
            path (str): Path to save the best model.
            verbose (bool): If True, prints updates.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = - val_loss_min if val_loss_min < float("inf") else None
        self.early_stop = False
        self.val_loss_min = val_loss_min

    def __call__(self, val_loss, epoch, val_accuracy, model):
        score = -val_loss  # Use negative loss for maximization

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, val_accuracy, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, val_accuracy, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        # torch.save(model.state_dict(), self.path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            self.path,
        )

        self.val_loss_min = val_loss


num_classes = len(np.unique(train_labels))
# class_weights = compute_class_weight(
#     "balanced", classes=np.unique(train_labels), y=train_labels
# )
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(
#     "mps"
#     if torch.backends.mps.is_available()
#     else "cuda" if torch.cuda.is_available() else "cpu"
# )
# print("Class Weights:", class_weights)

model = ResNetClassifier(num_classes)

# Define device, including support for macOS MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using macOS GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
    print(f"Using {torch.cuda.device_count()} GPUs")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Enable multi-GPU support
else:
    device = torch.device("cpu")
    print("Using CPU")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, labels):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(outputs, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(gamma=2, alpha=class_weights_tensor)
# criterion = FocalLoss(gamma=2)
# criterion = nn.CrossEntropyLoss()


# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
# Create a directory for saving checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', None)
    val_accuracy = checkpoint.get('val_accuracy', None)
    print(f"Resumed from epoch {start_epoch}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%")
    return model, optimizer, start_epoch, val_loss

def save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, path)

    print(f"Checkpoint saved at {path}")

def save_learning_curve(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curve")
    plt.savefig("learning_curve.png")
    plt.close()

def save_combined_plot(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Save the plot
    plt.savefig("learning_curve_combined.png")
    plt.close()

# Training loop with checkpointing
def train(model, train_loader, val_loader, optimizer, criterion, device, start_epoch,  val_loss, total_epochs, checkpoint_dir):
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=30, delta=0.001, path=os.path.join(checkpoint_dir, "best_model.pth"), verbose=True, val_loss_min=val_loss)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"{datetime.now()}: Epoch {epoch + 1}/{total_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # # Save checkpoint
        # checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_valloss_{val_loss:.4f}_valacc_{val_accuracy:.2f}.pth")
        # save_checkpoint(model, optimizer, epoch + 1, val_loss, val_accuracy, checkpoint_path)

        # Call EarlyStopping
        early_stopping(val_loss, epoch=epoch+1,  val_accuracy=val_accuracy, model=model)

        # Stop training if early stopping is triggered
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        #     save_checkpoint(model, optimizer, epoch + 1, val_loss, val_accuracy, best_model_path)            

        #     print(f"Best model saved at {best_model_path}")

    save_combined_plot(train_losses, val_losses, train_accuracies, val_accuracies)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


# Example: Start training from last checkpoint
checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
start_epoch = 0
val_loss = float("inf")
if os.path.exists(checkpoint_path):
    model, optimizer, start_epoch, val_loss = load_checkpoint(checkpoint_path, model, optimizer)


# Train the model
total_epochs = 150
train(model, train_loader, val_loader, optimizer, criterion, device, start_epoch, val_loss, total_epochs, checkpoint_dir)
