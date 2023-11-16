import math
import chess
from board import evaluate_board
from model import CheckMatrixModel
from transposition_table import TranspositionTable



class MCTSNode:
    def __init__(self, board: chess.Board, model: CheckMatrixModel, device, parent=None):
        self.board = board
        self.model = model
        self.device = device
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        self.transposition_table = TranspositionTable()
        self.board_value = self.evaluate_board()


    def evaluate_board(self):
        fen = self.board.fen()
        if self.transposition_table.lookup(fen):
            return self.transposition_table.lookup(fen)
        value = evaluate_board(self.board, self.model, self.device)
        self.transposition_table.store(fen, value)
        return value


    def select_child(self):
        # UCB1 = (win rate) + C * sqrt(ln(total visits) / child visits)
        C = 1.4  # Exploration parameter
        return max(self.children, key=lambda child: child.board_value + C * math.sqrt(math.log(self.visits + 1) / (child.visits + 1)))


    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        if self.transposition_table.lookup(new_board.fen()):
            return  # Skip if already in transposition table
        
        child_node = MCTSNode(new_board, self.model, self.device, parent=self)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node


    def update(self, result):
        self.visits += 1
        self.wins += result
