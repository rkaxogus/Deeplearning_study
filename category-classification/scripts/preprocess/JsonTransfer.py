import os
import shutil
import json

# Validation 데이터를 10개 카테고리로 재분류
source_dir = r"C:\Users\승승협\Desktop\캠핑용품데이터\정리된데이터_valid"
target_dir = r"C:\Users\승승협\Desktop\캠핑용품데이터\category_valid"

# 10개 카테고리 폴더 생성
categories = [
    "텐트_타프", "테이블_체어", "매트_침낭", "화로대_버너",
    "식기_주방", "가방_박스_웨건", "랜턴", "전기_전자기기",
    "냉_난방용품", "공구_기타"
]

# 각 카테고리에 해당하는 category_id 매핑
category_mapping = {
    "텐트_타프": [487, 488],
    "테이블_체어": [477, 482],
    "매트_침낭": [],
    "화로대_버너": [474, 484, 485, 489],
    "식기_주방": [475, 490, 493, 494],
    "가방_박스_웨건": [478, 481],
    "랜턴": [486, 491],
    "전기_전자기기": [479, 483],
    "냉_난방용품": [476],
    "공구_기타": [492, 495]  # 연소형모기향 포함
}

# 출력 폴더 생성
for category in categories:
    os.makedirs(os.path.join(target_dir, category), exist_ok=True)

# 이미지 재분류
for category_folder in os.listdir(source_dir):
    source_category_path = os.path.join(source_dir, category_folder)
    
    # category_XXX 폴더가 아닌 경우 스킵
    if not category_folder.startswith("category_"):
        continue
    
    # category_id 추출
    category_id = int(category_folder.split("_")[1])
    
    # 매핑에 따라 새로운 카테고리로 이동
    for target_category, ids in category_mapping.items():
        if category_id in ids:
            target_category_path = os.path.join(target_dir, target_category)
            for image_file in os.listdir(source_category_path):
                src_path = os.path.join(source_category_path, image_file)
                dest_path = os.path.join(target_category_path, image_file)
                
                # 이미지 이동
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} to {dest_path}")
