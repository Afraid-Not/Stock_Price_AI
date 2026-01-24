import pandas as pd
import glob
import os
import torch
from tqdm import tqdm
import sys

# 현재 디렉토리를 sys.path에 추가하여 r06_model을 임포트할 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from r06_model import MultiViewBERT, MODEL_NAME, PROJECTION_DIM, MultiViewDataset
    from transformers import AutoTokenizer
except ImportError:
    print("r06_model.py를 찾을 수 없거나 임포트할 수 없습니다. 모델 관련 기능은 건너뜁니다.")
    MultiViewBERT = None

NEWS_PATH = "D:/stock/_data/news"
MODEL_PATH = "D:/stock/_data/pseudo/models/refiner_checkpoint_ep2.pt"
SAVE_PATH = "D:/stock/_data/pseudo/news_total_refined.parquet"

def get_file_list(news_path):
    file_pattern = os.path.join(news_path, 'NewsResult_*.xlsx')
    file_list = glob.glob(file_pattern)
    print(f"{len(file_list)}개의 파일이 있습니다.")
    return sorted(file_list)

def process_dataframe(df):
    """
    단일 데이터프레임 전처리: 날짜 파싱 및 컬럼 병합
    """
    # 1. 뉴스 식별자 파싱
    if '뉴스 식별자' in df.columns:
        try:
            # 식별자에서 날짜 부분 추출 (예: ... .20220101...)
            # 에러 방지를 위해 문자열 변환 및 예외 처리
            split_data = df['뉴스 식별자'].astype(str).str.split('.')
            
            # 정상적으로 split된 경우만 처리 (길이가 2 이상인 경우)
            valid_mask = split_data.str.len() > 1
            
            if valid_mask.any():
                date_parts = split_data.loc[valid_mask].str[1]
                
                df.loc[valid_mask, 'year'] = date_parts.str[0:4]
                df.loc[valid_mask, 'month'] = date_parts.str[4:6]
                df.loc[valid_mask, 'day'] = date_parts.str[6:8]
                df.loc[valid_mask, 'hour'] = date_parts.str[8:10]
                df.loc[valid_mask, 'minute'] = date_parts.str[10:12]
                df.loc[valid_mask, 'second'] = date_parts.str[12:14]
        except Exception as e:
            print(f"날짜 파싱 중 에러 발생: {e}")

    # 2. 컬럼 병합 함수
    def merge_cols(row, columns):
        values = []
        for col in columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                values.append(str(row[col]).strip())
        return ' '.join(values)

    # 통합 분류 병합
    cat_cols = ['통합 분류1', '통합 분류2', '통합 분류3']
    df['combined_category'] = df.apply(lambda x: merge_cols(x, cat_cols), axis=1)

    # 사건/사고 분류 병합
    acc_cols = ['사건/사고 분류1', '사건/사고 분류2', '사건/사고 분류3']
    df['combined_accident'] = df.apply(lambda x: merge_cols(x, acc_cols), axis=1)

    # 3. 모델 입력을 위한 컬럼명 변경 (r06_model.py 호환)
    rename_map = {
        '인물': 'sorted_person',
        '위치': 'sorted_place',
        '기관': 'sorted_institute',
        '키워드': 'sorted_keyword',
        '특성추출(가중치순 상위 50개)': 'sorted_features'
    }
    df = df.rename(columns=rename_map)
    
    # 필요한 컬럼만 남기거나, 혹은 전체 유지
    # NaN 값 채우기 (모델 입력을 위해)
    target_cols = ['sorted_person', 'sorted_place', 'sorted_institute', 'sorted_keyword', 'sorted_features']
    for col in target_cols:
        if col not in df.columns:
            df[col] = "" # 컬럼이 없으면 빈 문자열로 생성
        else:
            df[col] = df[col].fillna("")

    return df

def run_model_refining(df, model_path):
    """
    학습된 모델을 사용하여 데이터 임베딩 생성 (Refining)
    """
    if MultiViewBERT is None or not os.path.exists(model_path):
        print("모델 파일을 찾을 수 없거나 라이브러리 로드 실패. 전처리만 진행합니다.")
        return df

    print(f"모델 로드 중: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 토크나이저 및 모델 초기화
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = ['[인물]', '[위치]', '[기관]', '[키워드]', '[특성]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    model = MultiViewBERT(MODEL_NAME, PROJECTION_DIM, len(tokenizer))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Refining (임베딩 생성) 시작...")
    
    # 데이터셋 및 데이터로더 준비
    # 임시 parquet 저장 (Dataset 클래스 재사용을 위해)
    temp_path = "temp_refining.parquet"
    df.to_parquet(temp_path)
    
    dataset = MultiViewDataset(temp_path, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            input_ids, attn_mask = [b.to(device) for b in batch]
            
            # 모델 출력: (batch, views, dim)
            # 여기서는 views들의 평균이나 concat을 사용할 수 있음.
            # MultiViewBERT의 출력은 projected (normalize된) 벡터임.
            embeddings = model(input_ids, attn_mask)
            
            # 5개 뷰의 임베딩을 평균내어 하나의 문서 벡터로 만듦 (예시)
            # 혹은 (batch, views * dim)으로 펼칠 수도 있음
            # 여기서는 평균을 사용하겠습니다.
            doc_embeddings = torch.mean(embeddings, dim=1) 
            all_embeddings.append(doc_embeddings.cpu().numpy())
    
    # 임베딩을 리스트로 변환하여 DataFrame에 추가
    import numpy as np
    concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # DataFrame에 추가 (리스트 형태 또는 개별 컬럼)
    # parquet 저장을 위해 리스트로 저장하는 것이 편함
    df['embedding'] = list(concatenated_embeddings)
    
    # 임시 파일 삭제
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return df

def main():
    file_list = get_file_list(NEWS_PATH)
    
    all_dfs = []
    
    # 1. 파일 읽기 및 전처리
    for file in tqdm(file_list, desc="Loading Excel Files"):
        try:
            # 엑셀 파일 로드
            df = pd.read_excel(file)
            
            # 전처리 (날짜 파싱, 컬럼 병합)
            df = process_dataframe(df)
            
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_dfs:
        print("처리할 데이터가 없습니다.")
        return

    # 2. 전체 데이터 병합
    print("데이터 병합 중...")
    total_df = pd.concat(all_dfs, ignore_index=True)
    print(f"전체 데이터 크기: {total_df.shape}")

    # 3. 모델 Refining (임베딩 생성)
    # 사용자가 원한 것이 모델을 돌리는 것이므로 수행
    if os.path.exists(MODEL_PATH):
        try:
            total_df = run_model_refining(total_df, MODEL_PATH)
        except Exception as e:
            print(f"모델 실행 중 오류 발생: {e}")
            print("전처리된 데이터만 저장합니다.")
    else:
        print(f"모델 파일이 없습니다: {MODEL_PATH}")

    # 4. 결과 저장
    print(f"결과 저장 중: {SAVE_PATH}")
    total_df.to_parquet(SAVE_PATH)
    print("완료!")

if __name__ == "__main__":
    main()
