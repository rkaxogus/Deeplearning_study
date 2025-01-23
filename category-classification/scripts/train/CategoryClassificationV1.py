import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision Training
from torchvision.models import EfficientNet_B0_Weights

def main():
    # 경로 설정
    train_dir = r"C:\Users\승승협\Desktop\캠핑용품데이터\category_train"
    valid_dir = r"C:\Users\승승협\Desktop\캠핑용품데이터\category_valid"

    # 하이퍼파라미터
    BATCH_SIZE = 128
    IMG_SIZE = (224, 224)
    EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 모델 정의
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # Mixed Precision Training용 GradScaler

    # 학습 루프
    best_valid_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f"\n=== Starting Epoch {epoch + 1}/{EPOCHS} ===")
        model.train()
        train_loss, train_correct = 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixed Precision Training
            with autocast():  # 최신 PyTorch에서는 autocast 사용
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

            progress = int((batch_idx + 1) / len(train_loader) * 100)
            print(f"Epoch {epoch + 1}/{EPOCHS}: {progress}% completed", end="\r")

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss, valid_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                valid_correct += (outputs.argmax(1) == labels).sum().item()

        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = valid_correct / len(valid_loader.dataset)

        print(f"\nEpoch {epoch + 1}/{EPOCHS} completed.")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}")

        # Save the best model
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'accuracy': valid_accuracy
            }, os.path.join("models", "best_camping_model.pth"))
            print(f"Best model saved with accuracy: {valid_accuracy:.4f}")

    print("\nTraining completed. Best model saved as best_camping_model.pth")

if __name__ == "__main__":
    main()