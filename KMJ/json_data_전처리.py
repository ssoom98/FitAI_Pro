import os
import json
import zipfile
import pandas as pd
import re

# 데이터 폴더 설정
base_folder = os.path.normpath(r"C:\minjun\project01\source\data\건강관리를 위한 음식 이미지")

# 1️⃣ ZIP 파일 압축 해제 (ZIP 내부 ZIP도 풀기 & 파일명 깨짐 방지)
zip_files = []
for folder in ["Training", "Validation"]:
    folder_path = os.path.join(base_folder, folder)
    if os.path.exists(folder_path):
        zip_files.extend(
            [(folder, os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith(".zip") and "[라벨]" in f])

total_files = len(zip_files)
if total_files == 0:
    print("🔍 '[라벨]'이 포함된 압축 파일이 없습니다. 작업을 종료합니다.")
else:
    print(f"📦 총 {total_files}개의 '[라벨]' ZIP 파일을 처리합니다.")

for idx, (folder, zip_path) in enumerate(zip_files, start=1):
    extract_folder = os.path.normpath(os.path.join(base_folder, "DATA", folder))  # Training / Validation 폴더별 압축 해제
    os.makedirs(extract_folder, exist_ok=True)

    zip_name = os.path.basename(zip_path)
    print(f"[{idx}/{total_files}] ⏳ {zip_name} 압축 해제 중... (해제 경로: {extract_folder})")

    if not zipfile.is_zipfile(zip_path):
        print(f"⚠️ {zip_name} 은(는) 손상된 ZIP 파일입니다. 건너뜁니다.")
        continue

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            try:
                # CP437 → CP949 변환 (Windows에서 압축된 경우 파일명 깨짐 방지)
                filename = member.filename.encode('cp437').decode('cp949', errors='ignore')
                safe_path = os.path.normpath(os.path.join(extract_folder, filename))
                if not safe_path.startswith(extract_folder):
                    print(f"🚨 경고: 위험한 경로 탐지 {filename}")
                    continue
                
                # 디렉토리인 경우 생성
                if member.is_dir():
                    os.makedirs(safe_path, exist_ok=True)
                else:
                    # 파일 저장
                    with open(safe_path, "wb") as f:
                        f.write(zip_ref.read(member.filename))
                        
            except OSError as e:
                print(f"⚠️ 파일 추출 중 오류 발생: {e}")
                continue
        print(f"✅ {zip_name} 압축 해제 완료 → {extract_folder}")

print("📂 모든 '[라벨]' ZIP 파일 해제 및 정리 완료!")

# 2️⃣ ZIP 내부에 또 ZIP이 있는 경우 자동 해제
def extract_nested_zips(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                print(f"🔄 내부 ZIP 해제: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)  # 같은 폴더에 압축 해제
                os.remove(zip_path)  # 내부 ZIP 삭제

extract_nested_zips(os.path.join(base_folder, "DATA"))
print("📂 모든 ZIP 파일 및 내부 ZIP 해제 완료!")

# 3️⃣ JSON → CSV 변환
json_files = []
extract_folder = os.path.normpath(os.path.join(base_folder, "DATA"))  # 압축 해제된 JSON 경로 통합
for root, _, files in os.walk(extract_folder):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))

print(f"🔍 총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

def extract_korean(text):
    return "".join(re.findall(r'[가-힣]+', text))  # 한글만 추출

all_data = []
for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 파일 로드 오류 발생: {json_path}, 오류: {e}")
            continue
    
    folder_name = os.path.basename(os.path.dirname(json_path))  # JSON 파일이 위치한 폴더명
    food_name_kor = extract_korean(folder_name)  # 한글만 추출

    for item in json_data:
        filtered_item = {
            "식품명(영문)": item.get("Name", "").strip(),  # JSON의 "Name" 필드를 "식품명(영문)"으로 사용
            "식품명": food_name_kor,  # 폴더명에서 한글만 추출
            "1회 섭취량(g)": item.get("Serving Size", "").strip(),
            "당류(g)": item.get("당류(g)", "")
        }

        if filtered_item["1회 섭취량(g)"] == "xx":  # "xx" 값 처리
            filtered_item["1회 섭취량(g)"] = None
        
        all_data.append(filtered_item)

# CSV 저장
if all_data:
    json_csv_path = os.path.join(base_folder, "json_food_data.csv")
    df_json = pd.DataFrame(all_data)
    df_json.to_csv(json_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ JSON 데이터를 CSV로 저장 완료: {json_csv_path}")
else:
    print("⚠️ 변환할 JSON 데이터가 없습니다. CSV 파일을 생성하지 않습니다.")
