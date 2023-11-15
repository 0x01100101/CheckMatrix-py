import chess
import chess.engine
import torch
import numpy as np
from model import CheckMatrixModel
from constants import PIECE_INDICES



def evaluate_board(board: chess.Board, model: CheckMatrixModel, device):
    """
    Evaluates a board state using the model
    """
    board_states_tensor = generate_board_states(board).to(device)
    prediction = model(board_states_tensor)
    return prediction.item()


def board_to_tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 6x8x8 tensor

    0: Pawns
    1: Knights
    2: Bishops
    3: Rooks
    4: Queens
    5: Kings

    White pieces are represented by 1, black pieces are represented by -1
    """
    
    board_tensor = np.zeros((6, 8, 8), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            color = 1 if piece.color == chess.WHITE else -1
            board_tensor[PIECE_INDICES[piece.symbol().upper()], i // 8, i % 8] = color
    
    return torch.tensor(board_tensor)


def generate_board_states(board: chess.Board):
    """
    Generates a 1x8x8x8 tensor representing the board state
    State 1: board, State 2: white attackable squares, State 3: black attackable squares
    """
    board_tensor = board_to_tensor(board)
    board_tensor = board_tensor.view(1, 6, 8, 8)

    white_attackable_squares_tensor = np.zeros((1, 8, 8), dtype=np.float32)
    black_attackable_squares_tensor = np.zeros((1, 8, 8), dtype=np.float32)

    for color in [chess.WHITE, chess.BLACK]:
        for square in chess.SQUARES:
            if board.is_attacked_by(color, square):
                row, col = square // 8, square % 8
                if color == chess.WHITE:
                    white_attackable_squares_tensor[0, row, col] = 1.0
                else:
                    black_attackable_squares_tensor[0, row, col] = 1.0

    white_attackable_squares_tensor = np.expand_dims(white_attackable_squares_tensor, axis=1)
    black_attackable_squares_tensor = np.expand_dims(black_attackable_squares_tensor, axis=1)

    board_states_tensor = np.concatenate((board_tensor, white_attackable_squares_tensor, black_attackable_squares_tensor), axis=1)
    board_states_tensor = torch.tensor(board_states_tensor, dtype=torch.float32)

    return board_states_tensor
