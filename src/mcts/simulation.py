import copy
import random
from logger import logger
from mcts.node import MCTSNode



def run_simulation(args):
    node, iterations = args
    local_node: MCTSNode = copy.deepcopy(node)
    results = []

    logger.debug(f"Running {iterations} iterations on worker")

    for _ in range(iterations):
        current_node: MCTSNode = local_node
        board_copy = local_node.board.copy()

        # Selection
        while current_node.untried_moves == [] and current_node.children != []:
            current_node = current_node.select_child()
            board_copy.push(current_node.board.move_stack[-1])

        # Expansion
        if current_node.untried_moves != []:
            move = random.choice(current_node.untried_moves)
            board_copy.push(move)
            current_node = current_node.add_child(move)

        # Simulation
        while not board_copy.is_game_over():
            board_copy.push(random.choice(list(board_copy.legal_moves)))

        # Backpropagation
        result = 1 if board_copy.result() == "1-0" else 0
        while current_node is not None:
            current_node.update(result)
            current_node = current_node.parent

        results.append((move, result))
    
    return results
