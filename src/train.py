import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os, json

TRAIN_DIR  = "data/train"
VAL_DIR    = "data/val"
MODEL_DIR  = "models"
EPOCHS     = 20
BATCH_SIZE = 32
LR         = 1e-3

os.makedirs(MODEL_DIR, exist_ok=True)

# Augmentations
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# Chargement
train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_set   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

print(f"Catégories : {train_set.classes}")
print(f"Train: {len(train_set)} | Val: {len(val_set)}")

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=0)

# Modèle ResNet-18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Entraînement
best_val_acc = 0.0
history = []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = train_correct = train_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (out.argmax(1) == labels).sum().item()
        train_total   += labels.size(0)

    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out       = model(imgs)
            val_loss += criterion(out, labels).item()
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    val_acc  = val_correct / val_total
    val_loss = val_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    history.append({
        "epoch": epoch+1,
        "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss,     "val_acc": val_acc,
    })

    # Sauvegarde meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   f"{MODEL_DIR}/best_model_resnet18.pth")
        print(f"  ✅ Meilleur modèle sauvegardé (val_acc={best_val_acc:.3f})")

# Sauvegarde métriques
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump({"best_val_acc": best_val_acc,
               "classes": train_set.classes,
               "history": history}, f, indent=2)

print(f"\n✅ Entraînement terminé ! Meilleure val_acc : {best_val_acc:.3f}")