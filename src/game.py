import chess
import chess.engine
import random
from logger import get_logger
from mcts.node import MCTSNode
from mcts.MCTS import MCTS
from board import generate_board_states
from reward import calculate_reward
from model import CheckMatrixModel, train_model
from config import load_config



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


def select_move(board: chess.Board, model: CheckMatrixModel, device, mcts_iterations=300, num_workers=6):
    root = MCTSNode(board, model, device)
    MCTS(root, iterations=mcts_iterations, num_workers=num_workers)
    best_move = max(root.children, key=lambda child: child.visits).board.peek()  # Select move with the highest visits
    return best_move


def play_game(model: CheckMatrixModel, optimizer, criterion, device, mcts_iterations, num_workers):
    board = chess.Board()
    is_ai_turn = random.choice([True, False])  # Randomize who starts
    board.turn = chess.WHITE if is_ai_turn else chess.BLACK

    while not board.is_game_over(claim_draw=True):
        move = select_move(board, model, device, mcts_iterations, num_workers)
        get_logger().debug(f"{'White' if board.turn == chess.WHITE else 'Black'} move: {move}")

        board.push(move)
        game_state = generate_board_states(board)
        turn_data = [(game_state, calculate_reward(board))]
        print(board)
        get_logger().debug(f"Board: {board.fen()}")

        train_model(model, turn_data, optimizer, criterion, device)

    result = board.result(claim_draw=True)
    return result
