"""
Pipeline d'entraînement complet pour Atelier 4 - Version 3.0
Classification d'images de produits sucrés avec ResNet-18
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Configuration ---
DATA_DIR = "data/raw"
CSV_FILE = os.path.join(DATA_DIR, "metadata_categories_180.csv")
IMG_DIR = os.path.join(DATA_DIR, "images", "categories")
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
SEED = 42

# --- Reproductibilité ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Dataset personnalisé ---
class SugarDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Regrouper les catégories similaires
        self.data['simple_category'] = self.data['category'].apply(self.simplify_category)

        # Catégories simplifiées
        self.categories = self.data['simple_category'].unique()
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

    def simplify_category(self, category):
        """Simplifier les catégories OpenFoodFacts en catégories générales"""
        if pd.isna(category):
            return 'other'

        category = category.lower()

        if 'white' in category or 'granulated' in category or 'caster' in category:
            return 'white_sugar'
        elif 'brown' in category or 'cassonade' in category:
            return 'brown_sugar'
        elif 'powdered' in category or 'icing' in category:
            return 'powdered_sugar'
        elif 'vanilla' in category:
            return 'vanilla_sugar'
        elif 'coconut' in category:
            return 'coconut_sugar'
        elif 'cane' in category and 'brown' not in category:
            return 'cane_sugar'
        elif 'beet' in category:
            return 'beet_sugar'
        elif 'chocolate' in category:
            return 'chocolate'
        else:
            return 'other_sugar'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['foodId']}.jpg")

        # Charger l'image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Image manquante, créer une image noire
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        label = self.category_to_idx[row['simple_category']]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Chargement du dataset ---
dataset = SugarDataset(CSV_FILE, IMG_DIR, transform=train_transform)
print(f"Dataset chargé: {len(dataset)} images")
print(f"Catégories: {dataset.categories}")

# --- Split train/val/test ---
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Appliquer les transformations correctes
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# --- Modèle ---
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights="IMAGENET1K_V1")

        # Fine-tuning complet
        for param in self.model.parameters():
            param.requires_grad = True

        # Remplacer la couche finale
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Entraînement ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de: {device}")

model = ResNetClassifier(len(dataset.categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Suivi des métriques
train_losses = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Entraînement
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
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

    # Métriques
    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Sauvegarder le meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model.pth')
        print("→ Meilleur modèle sauvegardé")

    scheduler.step()

print(f"Entraînement terminé. Meilleure précision validation: {best_val_acc:.4f}")

# --- Graphiques ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('reports/training_curves.png')
plt.show()

# --- Évaluation finale ---
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Rapport de classification
print("\nRapport de classification:")
print(classification_report(all_labels, all_preds, target_names=dataset.categories, labels=list(range(len(dataset.categories)))))

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.categories, yticklabels=dataset.categories, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png')
plt.show()

print("Pipeline d'entraînement terminé !")
print("Modèle sauvegardé dans: models/best_model.pth")
print("Rapports sauvegardés dans: reports/")