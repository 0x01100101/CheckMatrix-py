import chess
from constants import PIECE_VALUES



def calculate_control(board: chess.Board):
    """
    Calculate the control of the center squares for both players on a given chess board.
    """
    control = 0
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]

    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            control += 1 if square in center_squares else 0.5
        if board.is_attacked_by(chess.BLACK, square):
            control -= 1 if square in center_squares else 0.5

    return control


def calculate_piece_activity(board: chess.Board):
    """
    Calculate the activity of all pieces on a given chess board.
    """
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


def calculate_king_safety(board: chess.Board, color: chess.Color):
    """
    Calculate the safety of the king for a given color on a given chess board.
    """
    king_square = board.king(color)
    
    # Basic factors contributing to king's safety
    safety_score = 0

    # Check the number of friendly pieces around the king
    friendly_pieces_around = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            square = king_square + i + 8 * j
            if square in range(64) and board.color_at(square) == color:
                friendly_pieces_around += 1

    # Add points for each friendly piece surrounding the king
    safety_score += friendly_pieces_around * 10

    # Deduct points for open lines towards the king (e.g., no pawns in front)
    for direction in [8, -8, -1, 1]:  # up, down, left, right
        square = king_square + direction
        while square in range(64):
            if board.piece_at(square):
                if board.color_at(square) != color:
                    safety_score -= 5
                break
            square += direction

    return safety_score


def calculate_pawn_structure_score(board: chess.Board, color: chess.Color):
    """
    Calculate a score for the pawn structure for a given color on a chess board.
    """
    pawn_structure_score = 0

    # Iterate over all squares and evaluate pawns
    for square in range(64):
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            pawn_structure_score += evaluate_pawn_position(board, square)

    return pawn_structure_score


def evaluate_pawn_position(board: chess.Board, square: chess.Square):
    """
    Evaluate the position of a single pawn.
    """
    score = 0
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    # Check for doubled pawns
    if board.pieces(chess.PAWN, board.color_at(square)).tolist().count(file_index) > 1:
        score -= 5

    # Check for isolated pawns
    isolated = True
    for adjacent_file in [file_index - 1, file_index + 1]:
        if 0 <= adjacent_file <= 7:  # Ensure within board limits
            if board.pieces(chess.PAWN, board.color_at(square)).tolist().count(adjacent_file) > 0:
                isolated = False
                break
    if isolated:
        score -= 10

    # Check for passed pawns
    passed = True
    opponent_color = not board.color_at(square)
    for rank in range(rank_index + 1, 8):
        for file in range(8):
            if file == file_index and board.piece_at(chess.square(file, rank)) and board.piece_at(chess.square(file, rank)).color == opponent_color and board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN:
                passed = False
                break
        if not passed:
            break
    if passed:
        score += 20

    return score


def calculate_reward(board: chess.Board):
    material_balance = sum(PIECE_VALUES[piece.symbol().upper()] for piece in board.piece_map().values())

    control_balance = calculate_control(board)
    activity_balance = calculate_piece_activity(board)
    king_safty_balance = calculate_king_safety(board, chess.WHITE) - calculate_king_safety(board, chess.BLACK)
    pawn_structure_balance = calculate_pawn_structure_score(board, chess.WHITE) - calculate_pawn_structure_score(board, chess.BLACK)


    # Calculate reward
    reward = 0
    if board.is_checkmate():
        reward = 1.0
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        reward = 0.0
    else:
        reward = (material_balance + control_balance + activity_balance + king_safty_balance + pawn_structure_balance) / 39.0

    return reward
