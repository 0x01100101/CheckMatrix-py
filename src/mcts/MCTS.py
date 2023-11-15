from multiprocessing import Pool
from logger import logger
from mcts.node import MCTSNode
from mcts.simulation import run_simulation
import chess



def MCTS(root: MCTSNode, iterations=500, num_workers=4):
    try:
        pool = Pool(num_workers)
        logger.debug(f"Running MCTS with {num_workers} workers")
        worker_iterations = iterations // num_workers
        logger.debug(f"Running {worker_iterations} iterations per worker")
        tasks = [(root, worker_iterations) for _ in range(num_workers)]
        logger.debug(f"Running {len(tasks)} tasks")

        all_results = pool.map(run_simulation, tasks)
        logger.debug(f"Finished running {len(tasks)} tasks")
    except Exception as e:
        logger.error(f"An error occurred during MCTS: {e}")
    finally:
        pool.close()
        logger.debug("Closed pool")
        pool.terminate()
        logger.debug("Terminated pool")
        pool.join()
        logger.debug("Joined pool")

    
    # Aggregate results
    for worker_results in all_results:
        for move, result in worker_results:
            apply_move_result_to_tree(root, move, result)


def apply_move_result_to_tree(root: MCTSNode, move: chess.Move, result):
    if root.board.is_legal(move):
        node = find_or_create_node(root, move)
        if node:
            node.update(result)


def find_or_create_node(root: MCTSNode, move: chess.Move):
    for child in root.children:
        if child.board.peek() == move: # Check the last move of the child
            return child

    # If the node doesn't exist, create it
    new_board = root.board.copy()
    new_board.push(move)
    new_node = MCTSNode(new_board, root.model, root.device, parent=root)
    root.children.append(new_node)
    return new_node
