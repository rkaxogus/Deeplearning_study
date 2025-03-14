import os
import json
import pandas as pd

# 1) 엑셀 읽기
excel_path = r"C:\Users\user\OneDrive\Desktop\Resnet18-real\csv\텐트 데이터 라벨.xlsx"
df = pd.read_excel(excel_path)

# 2) 'Tent Name' 문자열로 변환 후 공백 제거
df["Tent Name"] = df["Tent Name"].astype(str).str.strip()

# 딕셔너리 생성: {TentName -> Label}
label_dict = dict(zip(df["Tent Name"], df["Label"]))

root_dir = r"C:\Users\user\OneDrive\Desktop\Resnet18-real\data\processed_data3"
results = {}

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    # 폴더 이름도 strip() 적용
    folder_name_stripped = folder_name.strip()

    if os.path.isdir(folder_path):
        if folder_name_stripped in label_dict:
            label_id = label_dict[folder_name_stripped]
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    # 기존: key = f"{folder_name}/{filename}"
                    # 수정: 파일명만 key 로 사용
                    key = filename
                    results[key] = label_id
        else:
            print(f"[주의] 엑셀 라벨에 '{folder_name}'(이)가 없습니다. (스킵)")
    else:
        continue

output_json_path = r"C:\Users\user\OneDrive\Desktop\Resnet18-real\json\tent_labels.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"완료! 결과가 '{output_json_path}' 에 저장되었습니다.")
