import torch
import torch.nn as nn

class MultiScaleEnsemble(nn.Module):
    def __init__(self, input_dim):
        super(MultiScaleEnsemble, self).__init__()
        # 1. Informer 60일/30일 (Transformer Encoder로 경량 구현)
        self.informer_60 = self._build_transformer(input_dim)
        self.informer_30 = self._build_transformer(input_dim)
        
        # 2. LSTM 10일
        self.lstm_10 = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.fc_lstm = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        
        # 3. Simple DL (MLP) 1일
        self.mlp_1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        self.fc_informer = nn.Linear(128, 2)
        # 초기 가중치 설정 (Informer들에 더 비중을 둠)
        self.weights = nn.Parameter(torch.tensor([0.3, 0.3, 0.25, 0.15]), requires_grad=False)

    def _build_transformer(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.1),  # Dropout 추가
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=128, 
                    nhead=8, 
                    batch_first=True,
                    dropout=0.1  # Transformer 내부에도 Dropout
                ), 
                num_layers=3
            )
        )

    def forward(self, x):
        # x shape: (batch, 60, input_dim)
        
        # Model 1: Informer 60 (Full window)
        out_60 = torch.softmax(self.fc_informer(self.informer_60(x)[:, -1, :]), dim=1)
        
        # Model 2: Informer 30 (Recent 30 days)
        out_30 = torch.softmax(self.fc_informer(self.informer_30(x[:, -30:, :])[:, -1, :]), dim=1)
        
        # Model 3: LSTM 10 (Recent 10 days)
        _, (h, _) = self.lstm_10(x[:, -10:, :])
        out_10 = torch.softmax(self.fc_lstm(h[-1]), dim=1)
        
        # Model 4: MLP 1 (Only Today)
        out_1 = torch.softmax(self.mlp_1(x[:, -1, :]), dim=1)
        
        # Weighted Soft Voting
        final_out = (out_60 * self.weights[0] + 
                     out_30 * self.weights[1] + 
                     out_10 * self.weights[2] + 
                     out_1 * self.weights[3])
        return final_out