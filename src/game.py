import chess
import chess.engine
import random
from logger import logger
from mcts.node import MCTSNode
from mcts.MCTS import MCTS
from board import generate_board_states
from reward import calculate_reward
from model import CheckMatrixModel, train_model



class Stockfish:
    def __init__(self) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")


    def get_move(self, board, time_limit=0.1):
        result = self.engine.play(board, chess.engine.Limit(time=time_limit, depth=1))
        return result.move

    def get_evaluation(self, board, time_limit=0.1):
        with self.engine.analysis(board, chess.engine.Limit(time=time_limit)) as analysis:
            for info in analysis:
                if info.get("score"):
                    score = info["score"].white().score(mate_score=1000)
                    return score
        return 0


def select_move(board: chess.Board, model: CheckMatrixModel, device, mcts_iterations=300, num_workers=6):
    root = MCTSNode(board, model, device)
    MCTS(root, iterations=mcts_iterations, num_workers=num_workers)
    best_move = max(root.children, key=lambda child: child.visits).board.peek()  # Select move with the highest visits
    return best_move


def play_game(model: CheckMatrixModel, optimizer, criterion, device):
    board = chess.Board()
    is_ai_turn = random.choice([True, False])  # Randomize who starts
    board.turn = chess.WHITE if is_ai_turn else chess.BLACK

    while not board.is_game_over(claim_draw=True):
        move = select_move(board, model, device)
        logger.debug(f"{"White" if board.turn == chess.WHITE else "Black"} move: {move}")

        board.push(move)
        game_state = generate_board_states(board)
        turn_data = [(game_state, calculate_reward(board))]
        print(board)
        logger.debug(f"Board: {board.fen()}")

        train_model(model, turn_data, optimizer, criterion, device)

    result = board.result(claim_draw=True)
    return result
