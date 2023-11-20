import chess
import chess.engine
import random
import torch.nn as nn
import torch.optim as optim
from logger import get_logger
from mcts.node import MCTSNode
from mcts.MCTS import MCTS
from board import generate_board_states
from reward import calculate_reward
from model import CheckMatrixModel, train_model
from config import Opponent, Device, load_config



class Stockfish:
    def __init__(self, path: str = "") -> None:
        self.config = load_config().stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(self.config.path or path)
        self.limit = chess.engine.Limit(time=self.config.time, depth=self.config.depth, nodes=self.config.nodes)


    def get_move(self, board):
        result = self.engine.play(board, self.limit)
        return result.move

    def get_evaluation(self, board):
        with self.engine.analysis(board, self.limit) as analysis:
            for info in analysis:
                if not info.get("score"):
                    continue
                
                score = info["score"].white().score(mate_score=1000)
                return score
        
        return 0


def select_move(model: CheckMatrixModel, board: chess.Board, device: Device, mcts_iterations=300, num_workers=6):
    root = MCTSNode(board, model, device)
    MCTS(root, iterations=mcts_iterations, num_workers=num_workers)
    best_move = max(root.children, key=lambda child: child.visits).board.peek()  # Select move with the highest visits
    return best_move


def validate_move(board: chess.Board, move: chess.Move):
    if move not in board.legal_moves:
        raise ValueError(f"Move {move} is not legal on board {board}")

def make_move(
        model: CheckMatrixModel, 
        board: chess.Board,
        device: Device,
        mcts_iterations: int,
        num_workers: int,
        opponent = Opponent.SELF,
        stockfish: Stockfish = None
    ):
    move = None

    match opponent:
        case Opponent.SELF:
            move = select_move(model, board, device, mcts_iterations, num_workers)
        case Opponent.STOCKFISH:
            move = stockfish.get_move(board)
        case Opponent.USER:
            move = input("Enter move: ")
        case Opponent.MIXED:
            if random.random() < 0.5:
                move = select_move(model, board, device, mcts_iterations, num_workers)
            else:
                move = stockfish.get_move(board)

    if opponent == Opponent.USER:
        try:
            move = chess.Move.from_uci(move)
            validate_move(board, move)
        except (chess.InvalidMoveError, ValueError) as e:
            get_logger().error(f"Invalid move: {e}")
            print("Invalid move, try again")
            return make_move(model, board, device, mcts_iterations, num_workers, opponent, stockfish)

    return move


def play_game(
        model: CheckMatrixModel,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: Device,
        mcts_iterations: int,
        num_workers: int,
        opponent = Opponent.SELF,
        stockfish: Stockfish = None
    ):
    board = chess.Board()
    is_ai_turn = random.choice([True, False])  # Randomize who starts
    board.turn = chess.WHITE if is_ai_turn else chess.BLACK

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            move = select_move(model, board, device, mcts_iterations, num_workers)
        else:
            move = make_move(model, board, device, mcts_iterations, num_workers, opponent, stockfish)

        get_logger().debug(f"{'White' if board.turn == chess.WHITE else 'Black'} move: {move}")

        board.push(move)
        game_state = generate_board_states(board)
        turn_data = [(game_state, calculate_reward(board))]
        print(board)
        get_logger().debug(f"Board: {board.fen()}")

        train_model(model, turn_data, optimizer, criterion, device)

    result = board.result(claim_draw=True)
    return result
