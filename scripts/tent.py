import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


# 브랜드와 텐트 이름
data = {
    "스노우피크": ["엔트리", "랜드락", "어메니티돔", "볼트", "헥스브리즈", "도크돔", "그랑베르크"],
    "코오롱스포츠": ["하이브라이트", "하이브라이트 TP", "하이브라이트 더블", "하이랜드 쉘터", "슈퍼팰리스"],
    "헬리녹스": ["노나돔", "알파돔", "필드타프", "바람타플", "발토"],
    "헬스포츠": ["켐피", "파세스", "리손티엘", "비르세"],
    "힐레베르그": ["아크투스", "케로", "나모스", "닉", "우나", "카이텀", "알락", "스타이카"]
}

# 기본 디렉토리 경로 설정
base_dir = os.path.join("C:/Users/user/OneDrive/Desktop/캠터/캠터감태현", "tent_data")

# 디렉토리 생성
for brand, tents in data.items():
    brand_path = os.path.join(base_dir, brand)
    os.makedirs(brand_path, exist_ok=True)  # 브랜드 폴더 생성
    
    for tent in tents:
        tent_path = os.path.join(brand_path, tent)
        os.makedirs(tent_path, exist_ok=True)  # 텐트 이름 폴더 생성

print(f"{base_dir} 디렉토리 구조가 성공적으로 생성되었습니다.")

# 현재 작업 디렉토리 출력
print(f"현재 작업 디렉토리: {os.getcwd()}")
