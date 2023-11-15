import chess
from constants import PIECE_VALUES



def calculate_control(board: chess.Board):
    control = 0
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]

    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            control += 1 if square in center_squares else 0.5
        if board.is_attacked_by(chess.BLACK, square):
            control -= 1 if square in center_squares else 0.5

    return control


def calculate_piece_activity(board: chess.Board):
    activity_score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = 0
            if piece.piece_type == chess.PAWN:
                value = 0.1
            elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                value = 0.3
            elif piece.piece_type == chess.ROOK:
                value = 0.5
            elif piece.piece_type == chess.QUEEN:
                value = 0.9

            legal_moves_count = sum(1 for move in board.legal_moves if move.from_square == square)
            activity_score += value * legal_moves_count

    return activity_score


def calculate_reward(board: chess.Board):
    material_balance = sum(PIECE_VALUES[piece.symbol().upper()] for piece in board.piece_map().values())

    control_balance = calculate_control(board)
    activity_balance = calculate_piece_activity(board)

    reward = 0
    if board.is_checkmate():
        reward = 1.0
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        reward = 0.0
    else:
        reward = (material_balance + control_balance + activity_balance) / 39.0

    return reward
