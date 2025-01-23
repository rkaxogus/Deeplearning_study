import os
import shutil

# 기존 경로 설정
base_dir = "C:/Users/user/OneDrive/Desktop/CAMTER/CAMTERKAM/tent_data"

# 변경된 데이터 저장 경로
new_base_dir = "C:/Users/user/OneDrive/Desktop/CAMTER/CAMTERKAM/tent_data_by_product"
os.makedirs(new_base_dir, exist_ok=True)  # 새로운 폴더 생성

# 기존 경로에서 상품 폴더를 최상위 폴더로 이동
for brand in os.listdir(base_dir):
    brand_path = os.path.join(base_dir, brand)
    
    if os.path.isdir(brand_path):  # 브랜드 폴더인지 확인
        for product in os.listdir(brand_path):
            product_path = os.path.join(brand_path, product)
            
            if os.path.isdir(product_path):  # 상품 폴더인지 확인
                # 상품 폴더를 새로운 최상위 폴더로 이동
                new_product_path = os.path.join(new_base_dir, product)
                shutil.move(product_path, new_product_path)

# 기존 브랜드 폴더 삭제
for brand in os.listdir(base_dir):
    brand_path = os.path.join(base_dir, brand)
    if os.path.isdir(brand_path):
        os.rmdir(brand_path)  # 브랜드 폴더 삭제

print(f"데이터 구조가 '{new_base_dir}'로 성공적으로 변경되었습니다!")
