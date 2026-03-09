import torch
import torch.nn as nn



class LiSANet(nn.Module):
    def __init__(self, input_channels=8, gru_hidden_size=256, num_gru_layers=2, mic_embedding_dim=16):
        super(LiSANet, self).__init__()
        
        # ENCODER GEOMETRIA MICROFONI
        self.mic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, mic_embedding_dim),
            nn.ReLU()
        )
        
        # CNN BACKBONE
        # Obiettivo: Ridurre la frequenza ma mantenere il Tempo (fondamentale per la fase/TDOA)
        self.cnn_backbone = nn.Sequential(
            # Block 1: 8 -> 32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Pooling Asimmetrico: (2, 1) dimezza Freq, mantiene Tempo
            nn.MaxPool2d(kernel_size=(2, 1)), 
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            # Output: (Batch, 256, 1, Time) per ogni blocco (Frame temporale T)
            nn.AdaptiveAvgPool2d((1, None)), 
        )
                
        # GRU
        cnn_out_dim = 13824
        self.gru = nn.GRU(
            input_size=cnn_out_dim + mic_embedding_dim,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.2 if num_gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # SHARED HEAD
        self.shared_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # HEAD DISTANZA
        self.dist_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        # HEAD ACCDOA (Attività e Angolo combinati: [prob*cos, prob*sin])
        self.accdoa_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, mic_coords, hidden_state=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # 1. Mic Geometry Embedding
        mic_embedding = self.mic_encoder(mic_coords)
        
        # 2. CNN Features (TimeDistributed Logica Integrata)
        x_reshape = x.contiguous().view(batch_size * seq_len, *x.shape[2:]) 
        features = self.cnn_backbone(x_reshape)
        features = features.view(batch_size, seq_len, -1)
        
        # 3. Concatenation
        mic_embedding_expanded = mic_embedding.unsqueeze(1).expand(batch_size, seq_len, -1)
        combined_features = torch.cat([features, mic_embedding_expanded], dim=-1)
        
        # 4. GRU Sequence Modeling
        rnn_out, new_hidden_state = self.gru(combined_features, hidden_state)
        
        # 5. Heads
        shared_features = self.shared_head(rnn_out)
        
        # Inversa distanza normalizzata (output lineare: può essere negativo)
        dist_pred = self.dist_head(shared_features).squeeze(-1)

        # ACCDOA: (Batches, SeqLen, 2)
        # L'output è un vettore [u, v] dove la norma è la probabilità (attività)
        # e l'angolo arctan2(v, u) è il DoA.
        accdoa_pred = self.accdoa_head(shared_features)
        
        return dist_pred, accdoa_pred, new_hidden_state

class LiSALSTMNet(nn.Module):
    def __init__(self, input_channels=8, lstm_hidden_size=256, num_lstm_layers=2, mic_embedding_dim=16):
        super(LiSALSTMNet, self).__init__()
        
        self.mic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, mic_embedding_dim),
            nn.ReLU()
        )
        
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, None)), 
        )
                
        cnn_out_dim = 13824
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim + mic_embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.shared_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.dist_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        self.accdoa_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, mic_coords, hidden_state=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        mic_embedding = self.mic_encoder(mic_coords)
        
        x_reshape = x.contiguous().view(batch_size * seq_len, *x.shape[2:]) 
        features = self.cnn_backbone(x_reshape)
        features = features.view(batch_size, seq_len, -1)
        
        mic_embedding_expanded = mic_embedding.unsqueeze(1).expand(batch_size, seq_len, -1)
        combined_features = torch.cat([features, mic_embedding_expanded], dim=-1)
        
        rnn_out, new_hidden_state = self.lstm(combined_features, hidden_state)
        
        shared_features = self.shared_head(rnn_out)
        
        # Inversa distanza normalizzata (output lineare: può essere negativo)
        dist_pred = self.dist_head(shared_features).squeeze(-1)

        accdoa_pred = self.accdoa_head(shared_features)
        
        return dist_pred, accdoa_pred, new_hidden_state

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    model_gru = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2)
    print(f"LiSANet (GRU) Total Trainable Parameters: {count_parameters(model_gru)}")
    
    model_lstm = LiSALSTMNet(input_channels=8, lstm_hidden_size=256, num_lstm_layers=2)
    print(f"LiSALSTMNet (LSTM) Total Trainable Parameters: {count_parameters(model_lstm)}")

if __name__ == "__main__":
    main()