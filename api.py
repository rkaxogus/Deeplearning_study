from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import zipfile
import tempfile
import os
import json

# FastAPI 앱 생성
app = FastAPI()

# GPU 또는 CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 불러오기
model_path = "C:/Users/user/OneDrive/Desktop/Resnet18-real/modelnew/resnet18-3.pth"
state_dict = torch.load(model_path, map_location=device)

# 모델의 출력층 크기 확인
num_classes = state_dict['fc.weight'].shape[0]

# 모델 정의 및 가중치 로드
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 이미지 변환 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ ZIP 파일 업로드 + 정답 레이블 검증 API (이모티콘 및 간결한 출력)
@app.post("/predict_tent/")
async def evaluate_model(file: UploadFile = File(...), labels: str = Form(...)):
    """
    ZIP 파일을 업로드하고, 정답 레이블을 비교하여 모델 정확도를 평가하는 API.
    
    :param file: 업로드된 ZIP 파일 (이미지 파일 포함)
    :param labels: JSON 형식의 정답 레이블 (예: '{"image1.jpg": 3, "image2.jpg": 7}')
    :return: 간략한 이미지별 결과와 전체 정답 개수 및 정확도
    """
    results = []
    correct_count = 0  # 맞춘 이미지 개수
    total_count = 0    # 전체 이미지 개수

    # JSON 형태의 정답 레이블 파싱 (예외 처리 포함)
    try:
        true_labels = json.loads(labels)
    except json.JSONDecodeError:
        return {"error": "정답 레이블이 올바른 JSON 형식이 아닙니다."}

    # 임시 폴더 생성 후 ZIP 파일 저장 및 압축 해제
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # 모든 하위 디렉토리 포함하여 이미지 파일 검색
        for root, dirs, files in os.walk(temp_dir):
            for img_name in files:
                if img_name.lower().endswith(("png", "jpg", "jpeg")):
                    img_path = os.path.join(root, img_name)
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue

                    image = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(image)
                        predicted_class = torch.argmax(output, 1).item()

                    true_class = true_labels.get(img_name, -1)
                    is_correct = (predicted_class == true_class)
                    emoji = "✅" if is_correct else "❌"
                    if is_correct:
                        correct_count += 1
                    total_count += 1

                    # 한 줄로 간단하게 결과 표현 (📄: 이미지, ✅/❌: 정답 여부)
                    results.append(f"📄 {img_name} : {emoji} (예측: {predicted_class}, 정답: {true_class if true_class != -1 else 'N/A'})")

    # 전체 정확도 및 정답 개수 계산
    accuracy = round((correct_count / total_count) * 100, 2) if total_count > 0 else 0.0
    summary = f"정답: {correct_count}/{total_count} (정확도: {accuracy}%)"

    return {
        "summary": summary,
        "results": results
    }

# FastAPI 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#http://127.0.0.1:8000/docs