import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer, get_linear_schedule_with_warmup
# AdamW는 이제 torch.optim에서 가져옵니다.
from torch.optim import AdamW
from tqdm import tqdm

# ==========================================
# 1. 설정 및 하이퍼파라미터
# ==========================================
DATA_PATH = r"D:\stock\_data\pseudo\news_total_sorted.parquet"
MODEL_SAVE_PATH = r"D:\stock\_data\pseudo\models"
RESUME_CHECKPOINT = r"D:\stock\_data\pseudo\models\refiner_checkpoint_ep1.pt" # 체크포인트 경로 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "klue/bert-base"
MAX_LEN = 128
BATCH_SIZE =  32
EPOCHS = 3
LEARNING_RATE = 2e-5
PROJECTION_DIM = 256
TEMPERATURE = 0.07

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# ==========================================
# 2. 커스텀 데이터셋 설계
# ==========================================
class MultiViewDataset(Dataset):
    def __init__(self, parquet_path, tokenizer):
        print("📖 데이터를 로드 중입니다. 잠시만 기다려 주세요...")
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.categories = ['sorted_person', 'sorted_place', 'sorted_institute', 'sorted_keyword', 'sorted_features']
        self.prefixes = ['[인물]', '[위치]', '[기관]', '[키워드]', '[특성]']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        view_input_ids = []
        view_attn_masks = []
        
        for cat, pref in zip(self.categories, self.prefixes):
            text = f"{pref} {row[cat]}"
            enc = self.tokenizer(
                text,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            view_input_ids.append(enc['input_ids'])      
            view_attn_masks.append(enc['attention_mask']) 
            
        return torch.cat(view_input_ids, dim=0), torch.cat(view_attn_masks, dim=0)

# ==========================================
# 3. Shared Encoder 모델 설계
# ==========================================
class MultiViewBERT(nn.Module):
    def __init__(self, model_name, projection_dim, special_tokens_len):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(special_tokens_len)
        
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        b, v, s = input_ids.shape
        flat_ids = input_ids.view(-1, s)
        flat_mask = attention_mask.view(-1, s)
        
        outputs = self.bert(flat_ids, attention_mask=flat_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        projected = self.projection(cls_output)
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected.view(b, v, -1)

# ==========================================
# 4. InfoNCE 손실 함수
# ==========================================
def compute_multiview_loss(embeddings, tau=0.07):
    b, v, d = embeddings.shape
    total_loss = 0
    pair_count = 0
    
    for i in range(v):
        for j in range(i + 1, v):
            view_i = embeddings[:, i, :] 
            view_j = embeddings[:, j, :] 
            
            logits = torch.matmul(view_i, view_j.T) / tau
            labels = torch.arange(b).to(embeddings.device)
            
            loss_i = F.cross_entropy(logits, labels)
            loss_j = F.cross_entropy(logits.T, labels)
            
            total_loss += (loss_i + loss_j) / 2
            pair_count += 1
            
    return total_loss / pair_count

# ==========================================
# 5. 메인 학습 루프
# ==========================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = ['[인물]', '[위치]', '[기관]', '[키워드]', '[특성]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    dataset = MultiViewDataset(DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiViewBERT(MODEL_NAME, PROJECTION_DIM, len(tokenizer)).to(DEVICE)
    
    # 수정된 부분: torch.optim에서 AdamW를 사용합니다.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    start_epoch = 0
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"🔄 체크포인트 로드 중: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ 학습 재개: Epoch {start_epoch+1} 부터 시작합니다.")
    else:
        print("⚠️ 지정된 체크포인트 파일이 없습니다. 처음부터 학습을 시작합니다.")
    
    num_training_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps
    )
    
    model.train()
    print(f"🔥 학습 시작 (Device: {DEVICE})")
    
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            input_ids, attn_mask = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            view_embeddings = model(input_ids, attn_mask)
            loss = compute_multiview_loss(view_embeddings, tau=TEMPERATURE)
            
            loss.backward()
            optimizer.step()
            scheduler.step() # 스케줄러 업데이트 추가
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} 종료 - 평균 Loss: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(MODEL_SAVE_PATH, f"refiner_checkpoint_ep{epoch+1}.pt"))

if __name__ == "__main__":
    main()