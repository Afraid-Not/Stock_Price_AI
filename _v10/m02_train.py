"""
학습 파이프라인 - 데이터 로드부터 모델 학습까지
"""
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor
from m01_model import StockPredictionModel, ModelEvaluator


class TrainingPipeline:
    """전체 학습 파이프라인"""
    
    def __init__(self, data_dir="D:/stock/_v10/_data/stock", output_dir="D:/stock/_v10"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.preprocessed_dir = os.path.join(output_dir, "_preprocessed")
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
    def preprocess_all_stocks(self):
        """모든 주식 데이터 전처리"""
        print("=" * 60)
        print("📁 데이터 전처리 시작")
        print("=" * 60)
        
        stock_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        processed_files = []
        
        for i, file_path in enumerate(stock_files):
            filename = os.path.basename(file_path)
            stock_code = filename.split("_")[0]
            
            print(f"\n[{i+1}/{len(stock_files)}] {stock_code} 처리 중...")
            
            # 1. 컬럼명 변환
            renamed_file = os.path.join(self.preprocessed_dir, f"{stock_code}_renamed.csv")
            if not os.path.exists(renamed_file):
                if not rename_file(file_path, renamed_file):
                    print(f"  ⚠️ {stock_code} 리네임 실패, 건너뜀")
                    continue
            
            # 2. 전처리
            final_file = os.path.join(self.preprocessed_dir, f"{stock_code}_final.csv")
            if not os.path.exists(final_file):
                try:
                    preprocessor = StockPreprocessor(stock_code=stock_code)
                    preprocessor.run_pipeline(renamed_file, final_file, is_train=True)
                except Exception as e:
                    print(f"  ⚠️ {stock_code} 전처리 실패: {e}")
                    continue
            
            processed_files.append(final_file)
            print(f"  ✅ {stock_code} 완료")
        
        print(f"\n✅ 총 {len(processed_files)}개 종목 전처리 완료")
        return processed_files
    
    def load_and_merge_data(self, file_list=None, min_rows=100):
        """전처리된 데이터 로드 및 병합"""
        if file_list is None:
            file_list = glob.glob(os.path.join(self.preprocessed_dir, "*_final.csv"))
        
        print(f"\n📊 {len(file_list)}개 파일 로드 중...")
        
        all_data = []
        for file_path in file_list:
            try:
                df = pd.read_csv(file_path)
                if len(df) < min_rows:
                    print(f"  ⚠️ {os.path.basename(file_path)}: 데이터 부족 ({len(df)}행), 건너뜀")
                    continue
                
                # 종목 코드 추가
                stock_code = os.path.basename(file_path).split("_")[0]
                df['stock_code'] = stock_code
                all_data.append(df)
                
            except Exception as e:
                print(f"  ⚠️ {file_path} 로드 실패: {e}")
        
        if not all_data:
            raise ValueError("로드된 데이터가 없습니다!")
        
        # 병합
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ 총 {len(merged_df)}행 데이터 로드 완료")
        
        return merged_df
    
    def prepare_features(self, df):
        """피처와 타겟 분리"""
        # 제외할 컬럼
        exclude_cols = ['날짜', 'target', 'next_rtn', 'stock_code']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # NaN 처리
        X = X.fillna(0)
        
        print(f"📋 피처 수: {len(feature_cols)}")
        print(f"📋 샘플 수: {len(X)}")
        print(f"📋 클래스 분포: 상승 {y.sum()} ({y.mean()*100:.1f}%), 하락 {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
        
        return X, y, feature_cols
    
    def train_model(self, X, y, test_size=0.2):
        """모델 학습"""
        # 시계열 데이터이므로 shuffle=False
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n📊 학습 데이터: {len(X_train)}행")
        print(f"📊 테스트 데이터: {len(X_test)}행")
        
        # 모델 학습
        model = StockPredictionModel()
        train_results = model.train(X_train, y_train)
        
        # 테스트 평가
        print("\n" + "=" * 60)
        print("🧪 테스트 데이터 평가")
        print("=" * 60)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        test_metrics = ModelEvaluator.evaluate(y_test, y_pred, y_proba)
        
        print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"   Precision: {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   F1 Score:  {test_metrics['f1']:.4f}")
        print(f"   AUC:       {test_metrics['auc']:.4f}")
        
        # 피처 중요도 출력
        print("\n📊 Top 10 중요 피처:")
        print(model.feature_importance.head(10).to_string(index=False))
        
        # 모델 저장
        model.save()
        
        return model, test_metrics
    
    def run_full_pipeline(self, preprocess=True):
        """전체 파이프라인 실행"""
        print("\n" + "🚀" * 20)
        print("      주가 예측 AI 학습 파이프라인")
        print("🚀" * 20 + "\n")
        
        # 1. 전처리
        if preprocess:
            self.preprocess_all_stocks()
        
        # 2. 데이터 로드
        df = self.load_and_merge_data()
        
        # 3. 피처 준비
        X, y, feature_cols = self.prepare_features(df)
        
        # 4. 모델 학습
        model, metrics = self.train_model(X, y)
        
        return model, metrics


def train_single_stock(stock_code, data_dir="D:/stock/_v10/_data/stock"):
    """단일 종목 학습"""
    stock_code = str(stock_code).zfill(6)
    
    # 파일 찾기
    pattern = os.path.join(data_dir, f"{stock_code}_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ {stock_code} 데이터 파일을 찾을 수 없습니다.")
        return None
    
    file_path = files[0]
    print(f"📂 파일: {file_path}")
    
    # 전처리
    preprocessor = StockPreprocessor(stock_code=stock_code)
    renamed_file = file_path.replace(".csv", "_renamed.csv")
    final_file = file_path.replace(".csv", "_final.csv")
    
    from s02_rename import rename_file
    rename_file(file_path, renamed_file)
    df = preprocessor.run_pipeline(renamed_file, final_file, is_train=True)
    
    # 피처 준비
    exclude_cols = ['날짜', 'target', 'next_rtn']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    # 모델 학습
    model = StockPredictionModel(model_dir=f"D:/stock/_v10/models/{stock_code}")
    model.train(X, y)
    model.save(suffix=f"_{stock_code}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="주가 예측 모델 학습")
    parser.add_argument("--stock", type=str, help="단일 종목 코드 (예: 005930)")
    parser.add_argument("--all", action="store_true", help="전체 종목 학습")
    parser.add_argument("--no-preprocess", action="store_true", help="전처리 건너뛰기")
    
    args = parser.parse_args()
    
    if args.stock:
        train_single_stock(args.stock)
    else:
        pipeline = TrainingPipeline()
        pipeline.run_full_pipeline(preprocess=not args.no_preprocess)

