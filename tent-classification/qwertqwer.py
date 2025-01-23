import torch

# 저장된 가중치 로드
model_path = "C:/Users/user/OneDrive/Desktop/Camter_IAI/tent-classification/models/efficientnet_tent_model2.pth"
state_dict = torch.load(model_path, map_location="cpu")

# 키 확인
print("state_dict의 키:")
for key in state_dict.keys():
    print(key)
