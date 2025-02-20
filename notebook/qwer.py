#클래스별로 분배 잘됨, 모델이름 model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
import csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast  # 혼합 정밀도 학습

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["FUNCTORCH_COMPILE_DISABLE"] = "1"

# ✅ 하이퍼파라미터 설정
batch_size = 32
epochs = 1000
learning_rate = 0.001
early_stop_threshold = 0.99  # 얼리스탑 기준 (검증 정확도 99% 이상)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로 설정
train_dir = 'C:/Users/user/OneDrive/Desktop/Resnet182-real/data/category_data3'
json_dir = 'C:/Users/user/OneDrive/Desktop/Resnet182-real/jsonnew'
csv_dir = 'C:/Users/user/OneDrive/Desktop/Resnet182-real/csv'
model_dir = 'C:/Users/user/OneDrive/Desktop/Resnet182-real/model'

# 필요한 디렉터리가 없으면 생성
os.makedirs(json_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ✅ 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ 데이터셋 로드
dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# ✅ 데이터 분할 (7:2:1 비율)
train_data, temp_data = train_test_split(dataset.samples, test_size=0.3, stratify=dataset.targets)
val_data, test_data = train_test_split(temp_data, test_size=0.33, stratify=[item[1] for item in temp_data])

# ✅ JSON 파일로 데이터셋 저장
train_json_path = os.path.join(json_dir, "trainqwer.json")
val_json_path = os.path.join(json_dir, "valqwer.json")
test_json_path = os.path.join(json_dir, "testqwer.json")

with open(train_json_path, "w") as f:
    json.dump(train_data, f)
with open(val_json_path, "w") as f:
    json.dump(val_data, f)
with open(test_json_path, "w") as f:
    json.dump(test_data, f)

print(f"✅ 학습 데이터 JSON 저장 완료: {train_json_path}")
print(f"✅ 검증 데이터 JSON 저장 완료: {val_json_path}")
print(f"✅ 테스트 데이터 JSON 저장 완료: {test_json_path}")

# ✅ 데이터셋을 위한 커스텀 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, dataset, transform=None):
        self.data = data
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = self.dataset.loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# ✅ 학습, 검증, 테스트 데이터로더 생성
train_dataset = CustomDataset(train_data, dataset, transform)
val_dataset = CustomDataset(val_data, dataset, transform)
test_dataset = CustomDataset(test_data, dataset, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ✅ EfficientNet-B0 모델 정의
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset.classes))
model = model.to(device)

# ✅ 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ 학습률 스케줄러 설정
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

# ✅ 혼합 정밀도 학습 스케일러
scaler = GradScaler()

# ✅ 모델 성능을 평가하는 함수
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

# ✅ 학습 루프 추가
best_val_acc = 0.0
best_epoch = 0
best_model_weights = None
epoch_log = []

print("\n📢 Training Started! Logging Every Epoch:\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"📢 Epoch [{epoch+1}/{epochs}] 시작"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    # ✅ 검증 단계
    val_acc, val_loss = evaluate(model, val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_model_weights = model.state_dict()

    epoch_log.append([epoch + 1, train_acc, train_loss, val_acc, val_loss])

    print(f"   🎯 Train Accuracy: {train_acc:.4f} | 📉 Train Loss: {train_loss:.4f} "
        f"| 🎯 Valid Accuracy: {val_acc:.4f} | 📉 Valid Loss: {val_loss:.4f}")

    if val_acc >= early_stop_threshold:
        print(f"🚨 얼리스탑! 검증 정확도가 {val_acc:.4f}로 0.99에 도달하여 학습을 종료합니다.")
        break

    scheduler.step()

# ✅ 최상의 모델 저장
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), os.path.join(model_dir, 'model3new.pth'))

# ✅ 학습 기록을 CSV 파일로 저장
with open(os.path.join(csv_dir, 'training_log.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss'])
    writer.writerows(epoch_log)

# ✅ 최종 모델을 테스트 데이터로 평가
test_acc, test_loss = evaluate(model, test_loader)
print(f"🎯 Test Accuracy: {test_acc:.4f}, 📉 Test Loss: {test_loss:.4f}")
