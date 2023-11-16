import torch.nn as nn
import torch
import math
from logger import get_logger
from config import load_config



class CheckMatrixModel(nn.Module):
    def __init__(self, num_layers=4, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(CheckMatrixModel, self).__init__()
        self.board_encoder = nn.Linear(8, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder = nn.Linear(d_model * 64, 1) # Output a single value for board evaluation

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, height * width, channels)  # Reshape to [batch_size, 64, 8]
        x = self.board_encoder(x) # Output shape: [batch_size, 64, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # Output shape should remain [batch_size, 64, d_model]
        x = x.view(batch_size, -1) # Flatten to [batch_size, d_model * 64]

        # Ensure x is flattened to [batch_size, d_model * 64] before passing to decoder
        x = self.decoder(x) # Decoder expects [batch_size, d_model * 64]
        return torch.tanh(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

def train_model(model, training_data, optimizer, criterion, device):
    config = load_config().model
    get_logger().debug("Training model")

    model.train()
    for board_tensor, outcome in training_data:
        optimizer.zero_grad()
        board_tensor = board_tensor.to(device)  # move tensor to device
        prediction = model(board_tensor)
        outcome_tensor = torch.tensor([[outcome]], dtype=torch.float32).to(device)  # move tensor to device
        loss = criterion(prediction, outcome_tensor)
        loss.backward()
        get_logger().debug(f"Loss: {loss.item()}")
        optimizer.step()
        model.zero_grad()

        torch.save(model.state_dict(), config.path)

    get_logger().debug("Finished training model")
