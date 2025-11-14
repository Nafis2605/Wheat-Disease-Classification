import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set up logging
logging.basicConfig(level=logging.INFO)

# Select device dynamically
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Custom Dataset Class for Loading Data
class NumpyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path).astype('float32') / 255.0
        self.labels = np.load(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Transpose (C, H, W) for PyTorch (if needed)
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# MobileNetV2 Model
def create_mobilenetv2_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    # Modify the classifier for the current dataset
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# Training Function
def train_model():
    # Paths to dataset
    train_images_path = 'dataset_new/train_images.npy'
    train_labels_path = 'dataset_new/train_labels.npy'
    val_images_path = 'dataset_new/val_images.npy'
    val_labels_path = 'dataset_new/val_labels.npy'
    test_images_path = 'dataset_new/test_images.npy'
    test_labels_path = 'dataset_new/test_labels.npy'

    # Load datasets
    train_dataset = NumpyDataset(train_images_path, train_labels_path)
    val_dataset = NumpyDataset(val_images_path, val_labels_path)
    test_dataset = NumpyDataset(test_images_path, test_labels_path)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine number of classes
    num_classes = len(np.unique(np.load(train_labels_path)))
    logging.info(f"Number of Classes: {num_classes}")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(np.load(train_labels_path)),
        y=np.load(train_labels_path)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logging.info(f"Class Weights: {class_weights}")

    # Create model
    model = create_mobilenetv2_model(num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Training loop with early stopping
    epochs = 100
    patience = 30
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # Append training metrics
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        # Append validation metrics
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mobilenetv2.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    # Save final model
    torch.save(model.state_dict(), 'final_mobilenetv2.pth')
    logging.info("Final model saved.")

    # Save training history
    with open('mnv2_training_history.pkl', 'wb') as f:
        pickle.dump({'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': val_losses, 'val_acc': val_accs}, f)

    # Plot training and validation metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig('mnv2_loss_acc_plot.png')
    #plt.show()

    # Test the model
    class_names = [
        "Aphid",
        "BlackRust",
        "BrownHead",
        "FusariumHead",
        "Healthy",
        "LeafBlight",
        "Mildew",
        "Mite",
        "Septoria",
        "Smut",
        "StripeRust",
        "TanSpot"
    ]

    model.eval()
    # Predict and evaluate
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Get probabilities
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    overall_metrics = {
        "Accuracy": report["accuracy"],
        "Precision": np.mean([v["precision"] for k, v in report.items() if k in class_names]),
        "Recall": np.mean([v["recall"] for k, v in report.items() if k in class_names]),
        "F1-Score": np.mean([v["f1-score"] for k, v in report.items() if k in class_names]),
    }

    print("\n=== Overall Metrics ===")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== Per Class Metrics ===")
    for class_name in class_names:
        print(f"\nClass: {class_name}")
        for metric, value in report[class_name].items():
            print(f" {metric.capitalize()}: {value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix_mobilenet3.png')
    #plt.show()

    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('normalized_confusion_matrix_mobilenet3.png')
   #plt.show()

    # ROC Curve
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('roc_curve_mobilenet3.png')
    #plt.show()

    # Calculate Specificity
    specificity = []
    for i in range(len(class_names)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))

    print("\n=== Specificity Per Class ===")
    for class_name, spec in zip(class_names, specificity):
        print(f"{class_name}: {spec:.4f}")

    print("\n=== Overall Metrics Including Specificity ===")
    overall_metrics["Specificity"] = np.mean(specificity)
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")



# Train the model
train_model()


