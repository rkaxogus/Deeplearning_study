from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import zipfile
import tempfile
import os
import json

app = FastAPI()

# -----------------------------
# 1. 모델 로딩
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"C:\Users\user\OneDrive\Desktop\Resnet18-real\modelnew\resnet18-3.pth"  # 실제 모델 경로
state_dict = torch.load(model_path, map_location=device)

num_classes = state_dict['fc.weight'].shape[0]

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -----------------------------
# 2. 이미지 변환 함수
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 3. 라벨 매핑 파일 (JSON) 로딩
# -----------------------------
# 서버에 미리 label_mapping.json을 준비해 둔다.
label_mapping_path = r"C:\Users\user\OneDrive\Desktop\Resnet18-real\json\tent_labels.json"

with open(label_mapping_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)
    # 예: {"image1.jpg": 3, "image2.jpg": 7, ...}

# -----------------------------
# 4. API 엔드포인트
# -----------------------------
@app.post("/predict_tent_no_labels/")
async def evaluate_model(file: UploadFile = File(...)):
    """
    ZIP 파일만 업로드하고, 서버에 있는 라벨 매핑 정보를 사용하여
    예측 정확도를 계산하는 API.
    """
    results = []
    correct_count = 0
    total_count = 0

    # 4-1) 임시 폴더에 ZIP 저장 후 압축 해제
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # 4-2) 압축 해제된 이미지 파일들을 순회
        for root, dirs, files in os.walk(temp_dir):
            for img_name in files:
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, img_name)

                    # 1) 이미지 열기
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue

                    # 2) 전처리 & 모델 예측
                    image = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(image)
                        predicted_class = torch.argmax(output, 1).item()

                    # 3) 서버 내부 라벨 매핑에서 정답 찾기
                    #    여기서는 "파일명"만 key로 사용한다고 가정
                    #    (ex. "image1.jpg")
                    true_class = label_map.get(img_name, -1)

                    is_correct = (predicted_class == true_class)
                    if is_correct:
                        correct_count += 1
                    total_count += 1

                    # 결과 저장 (간단히)
                    results.append({
                        "filename": img_name,
                        "predicted_class": predicted_class,
                        "true_class": true_class,
                        "correct": is_correct
                    })

    # 4-3) 전체 정확도 계산
    accuracy = round((correct_count / total_count) * 100, 2) if total_count > 0 else 0.0

    return {
        "total_images": total_count,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "results": results
    }

# -----------------------------
# 5. 서버 실행 (직접 실행 시)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#http://127.0.0.1:8000/docs