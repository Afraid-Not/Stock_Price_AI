import pandas as pd
import numpy as np
import argparse
import joblib
import json
from pathlib import Path
from datetime import datetime
import lightgbm as lgb

class StockPredictor:
    def __init__(self, model_dir="D:/stock/_v6/models", scaler_dir="D:/stock/_v6/scalers", stock_code=None):
        self.base_model_dir = Path(model_dir)
        self.scaler_dir = Path(scaler_dir)
        self.stock_code = stock_code or "DEFAULT"
        
        # 종목코드별 모델 디렉토리 설정
        if stock_code:
            self.model_dir = self.base_model_dir / stock_code
        else:
            self.model_dir = self.base_model_dir
        
        # Target scaler 로드 (종목별)
        target_scaler_path = self.scaler_dir / f"{self.stock_code}_target_scaler.bin"
        if target_scaler_path.exists():
            self.target_scaler = joblib.load(target_scaler_path)
            print(f"✅ Target scaler loaded: {target_scaler_path}")
        else:
            # 종목별 스케일러가 없으면 DEFAULT 스케일러 시도
            default_scaler_path = self.scaler_dir / "DEFAULT_target_scaler.bin"
            if default_scaler_path.exists():
                self.target_scaler = joblib.load(default_scaler_path)
                print(f"⚠️ Using DEFAULT scaler: {default_scaler_path}")
            else:
                print(f"⚠️ Target scaler not found: {target_scaler_path}")
                self.target_scaler = None
        
        self.lgbm_model = None
        self.ensemble_weights = None
    
    def load_models(self, model_name=None, stock_code=None):
        """저장된 모델 로드"""
        # 종목코드가 제공되면 해당 폴더 사용, 없으면 초기화 시 설정된 종목코드 사용
        if stock_code:
            search_dir = self.base_model_dir / stock_code
        elif self.stock_code and self.stock_code != "DEFAULT":
            search_dir = self.base_model_dir / self.stock_code
        else:
            search_dir = self.model_dir
        
        if model_name is None:
            # 종목코드 폴더에서 모델 찾기
            lgbm_files = list(search_dir.glob("*_lgbm.txt"))
            
            if not lgbm_files:
                # 종목코드 폴더에 없으면 기본 폴더에서 찾기 (하위 호환성)
                if stock_code:
                    lgbm_files = list(self.base_model_dir.glob(f"{stock_code}_*_lgbm.txt"))
                else:
                    lgbm_files = list(self.base_model_dir.glob("*_lgbm.txt"))
                if lgbm_files:
                    print(f"⚠️ 종목코드 폴더에 모델이 없어 기본 폴더에서 찾았습니다.")
                    search_dir = self.base_model_dir
            
            if not lgbm_files:
                raise FileNotFoundError(f"No LGBM model found in {search_dir}")
            
            # 파일명에서 timestamp 추출하여 가장 최근 것 선택
            lgbm_files.sort(key=lambda x: x.stem.split('_')[-1], reverse=True)
            model_name = lgbm_files[0].stem.replace('_lgbm', '')
            print(f"📂 Using model: {model_name}")
            
            # 모델이 있는 디렉토리로 설정
            self.model_dir = lgbm_files[0].parent
        
        lgbm_path = self.model_dir / f"{model_name}_lgbm.txt"
        
        if not lgbm_path.exists():
            raise FileNotFoundError(f"LGBM model not found: {lgbm_path}")
        
        print(f"\n📂 Loading models...")
        print(f"   LGBM: {lgbm_path}")
        
        self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
        
        # 앙상블 가중치 로드
        weights_path = self.model_dir / f"{model_name}_weights.json"
        self.ensemble_weights = None
        if weights_path.exists():
            try:
                with open(weights_path, 'r') as f:
                    self.ensemble_weights = json.load(f)
                weight_str = f"LGBM={self.ensemble_weights['lgbm']:.3f}"
                print(f"✅ Ensemble weights loaded: {weight_str}")
            except Exception as e:
                print(f"⚠️ Failed to load ensemble weights: {e}")
                self.ensemble_weights = None
        else:
            print(f"⚠️ Ensemble weights not found: {weights_path}")
            print(f"   Using LGBM only")
        
        print(f"✅ Models loaded successfully")
    
    def load_data(self, data_path, original_data_path=None):
        """예측용 데이터 로드"""
        print(f"\n📊 Loading data: {data_path}")
        df = pd.read_csv(data_path)
        
        # Target 컬럼이 있으면 제거 (예측용이므로)
        if 'target' in df.columns:
            X = df.drop(columns=['target']).values
            print(f"   ⚠️ 'target' column removed for prediction")
        else:
            X = df.values
        
        print(f"   Data shape: {X.shape[0]} rows × {X.shape[1]} features")
        
        # 원본 데이터에서 현재 종가 가져오기
        current_price = None
        if original_data_path:
            try:
                original_df = pd.read_csv(original_data_path)
                if '종가' in original_df.columns:
                    current_price = original_df['종가'].iloc[-1]
                    print(f"   Current price from original data: {current_price:,.0f} KRW")
                elif 'stck_clpr' in original_df.columns:
                    current_price = original_df['stck_clpr'].iloc[-1]
                    print(f"   Current price from original data: {current_price:,.0f} KRW")
            except Exception as e:
                print(f"   ⚠️ Could not load original data: {e}")
        
        return X, df, current_price
    
    def predict(self, X):
        """예측 수행"""
        if self.lgbm_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        print(f"\n🔮 Making predictions...")
        
        # LGBM 예측
        lgbm_pred = self.lgbm_model.predict(X, num_iteration=self.lgbm_model.best_iteration)
        
        # 저장된 가중치가 있으면 사용, 없으면 기본 가중치
        if self.ensemble_weights:
            ensemble_pred = self.ensemble_weights['lgbm'] * lgbm_pred
            weight_str = f"LGBM={self.ensemble_weights['lgbm']:.3f}"
            print(f"   Using ensemble weights: {weight_str}")
        else:
            ensemble_pred = lgbm_pred
            print(f"   Using LGBM only")
        
        print(f"   Predictions completed")
        print(f"   Prediction range: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
        
        return ensemble_pred, lgbm_pred
    
    def inverse_transform(self, predictions):
        """스케일러 역변환 (원본 수익률로 복원)"""
        if self.target_scaler is not None:
            predictions_original = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            print(f"\n🔄 Inverse transform applied")
            print(f"   Scaled range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"   Original range: [{predictions_original.min():.4f}, {predictions_original.max():.4f}]")
            return predictions_original
        else:
            print(f"\n⚠️ Scaler not found, returning scaled values")
            return predictions
    
    def calculate_next_day_price(self, current_price, predicted_return, lgbm_pred_original):
        """예측 수익률로 다음날 종가 계산 및 오차 범위 계산"""
        if current_price is None:
            return None, None, None
        
        # 다음날 종가 = 현재 종가 * (1 + 예측 수익률)
        next_day_price = current_price * (1 + predicted_return)
        
        # 오차 범위 계산 (예측값의 ±5% 범위)
        min_return = lgbm_pred_original * 0.95
        max_return = lgbm_pred_original * 1.05
        
        # 오차 범위 종가
        min_price = current_price * (1 + min_return)
        max_price = current_price * (1 + max_return)
        
        return next_day_price, min_price, max_price
    
    def save_results(self, df, predictions_original, predictions_scaled, lgbm_pred_original, 
                     current_price=None, output_path=None):
        """예측 결과 저장"""
        result_df = df.copy()
        result_df['predicted_return'] = predictions_original
        result_df['predicted_return_scaled'] = predictions_scaled
        
        # 방향 예측 (양수=상승, 음수=하락)
        result_df['predicted_direction'] = (predictions_original > 0).astype(int)
        result_df['predicted_direction_label'] = result_df['predicted_direction'].map({0: 'Down', 1: 'Up'})
        
        # 다음날 종가 계산 (마지막 행만)
        if current_price is not None and len(result_df) > 0:
            last_idx = len(result_df) - 1
            next_price, min_price, max_price = self.calculate_next_day_price(
                current_price, 
                predictions_original[last_idx],
                lgbm_pred_original[last_idx]
            )
            
            result_df.loc[last_idx, 'current_price'] = current_price
            result_df.loc[last_idx, 'next_day_price'] = next_price
            result_df.loc[last_idx, 'price_range_min'] = min_price
            result_df.loc[last_idx, 'price_range_max'] = max_price
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 Results saved: {output_path}")
        
        return result_df
    
    def print_summary(self, predictions_original, current_price=None, 
                     lgbm_pred_original=None):
        """예측 결과 요약 출력"""
        print(f"\n{'='*60}")
        print(f"📈 Prediction Summary")
        print(f"{'='*60}")
        
        up_count = (predictions_original > 0).sum()
        down_count = (predictions_original <= 0).sum()
        total = len(predictions_original)
        
        print(f"Total predictions: {total}")
        print(f"  Up (positive):   {up_count} ({up_count/total*100:.1f}%)")
        print(f"  Down (negative):  {down_count} ({down_count/total*100:.1f}%)")
        print(f"\nReturn statistics:")
        print(f"  Mean:   {predictions_original.mean():.6f}")
        print(f"  Std:    {predictions_original.std():.6f}")
        print(f"  Min:    {predictions_original.min():.6f}")
        print(f"  Max:    {predictions_original.max():.6f}")
        print(f"  Median: {np.median(predictions_original):.6f}")
        
        # 마지막 예측에 대한 다음날 종가 정보 출력
        if current_price is not None and len(predictions_original) > 0:
            last_pred = predictions_original[-1]
            next_price, min_price, max_price = self.calculate_next_day_price(
                current_price,
                last_pred,
                lgbm_pred_original[-1] if lgbm_pred_original is not None else last_pred
            )
            
            print(f"\n{'='*60}")
            print(f"💰 Next Day Price Prediction (Latest)")
            print(f"{'='*60}")
            print(f"Current Price:     {current_price:,.0f} KRW")
            print(f"Predicted Return:  {last_pred*100:.2f}%")
            print(f"\nNext Day Price:")
            print(f"  Expected:         {next_price:,.0f} KRW")
            print(f"  Price Range:      {min_price:,.0f} ~ {max_price:,.0f} KRW")
            print(f"  (Estimated based on model prediction variance)")

def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction using Ensemble Models")
    parser.add_argument("--data", type=str, required=True, help="Input data path (CSV)")
    parser.add_argument("--code", type=str, default=None, help="종목코드 (예: 005930, 해당 종목 모델/스케일러 자동 선택)")
    parser.add_argument("--original_data", type=str, default=None, help="Original data path to get current price (CSV)")
    parser.add_argument("--current_price", type=float, default=None, help="Current stock price (KRW)")
    parser.add_argument("--model_name", type=str, default=None, help="Model name (if None, uses latest)")
    parser.add_argument("--model_dir", type=str, default="D:/stock/_v6/models", help="Model directory")
    parser.add_argument("--scaler_dir", type=str, default="D:/stock/_v6/scalers", help="Scaler directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (if None, auto-generated)")
    
    args = parser.parse_args()
    
    # 종목코드 추출 (파일명에서 또는 직접 입력)
    stock_code = args.code
    if stock_code is None:
        # 파일명에서 추출 시도
        import re
        filename = Path(args.data).stem
        match = re.search(r'(\d{6})', filename)
        if match:
            stock_code = match.group(1)
            print(f"📌 종목코드 자동 추출: {stock_code}")
        else:
            stock_code = "DEFAULT"
            print(f"⚠️ 종목코드를 찾을 수 없습니다. DEFAULT 스케일러를 사용합니다.")
            print(f"   --code 옵션으로 직접 지정하세요.")
    
    # Predictor 초기화
    predictor = StockPredictor(
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
        stock_code=stock_code
    )
    
    # 모델 로드 (종목코드로 필터링)
    predictor.load_models(model_name=args.model_name, stock_code=stock_code)
    
    # 데이터 로드
    X, df, current_price_from_data = predictor.load_data(args.data, args.original_data)
    
    # 현재 종가 결정 (우선순위: 직접 입력 > 원본 데이터 > None)
    current_price = args.current_price if args.current_price is not None else current_price_from_data
    
    # 예측 수행
    ensemble_pred, lgbm_pred = predictor.predict(X)
    
    # 스케일러 역변환
    predictions_original = predictor.inverse_transform(ensemble_pred)
    lgbm_pred_original = predictor.inverse_transform(lgbm_pred)
    
    # 결과 요약 출력
    predictor.print_summary(predictions_original, current_price, lgbm_pred_original)
    
    # 결과 저장
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(args.data).parent / f"predictions_{timestamp}.csv"
    
    result_df = predictor.save_results(
        df, predictions_original, ensemble_pred, 
        lgbm_pred_original,
        current_price, args.output
    )
    
    print(f"\n✅ Prediction completed!")
    print(f"   Output: {args.output}")

if __name__ == "__main__":
    main()

