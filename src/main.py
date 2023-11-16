import torch
import torch.optim as optim
import os
import time
from logger import init_logger, get_logger
from model import CheckMatrixModel
from game import play_game
from config import load_config, Device
import argparse
from constants import CONFIG_PATH_ENV_VAR



def main():
    parser = argparse.ArgumentParser(description="CheckMatrix")
    parser.add_argument("--config", type=str, help="Path to config yaml file")

    args = parser.parse_args()


    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
    
        os.environ[CONFIG_PATH_ENV_VAR] = args.config


    init_logger()

    config = load_config().model
    
    device_type = config.device
    if device_type == Device.AUTO:
        device_type = Device.CUDA if torch.cuda.is_available() else Device.CPU

    device = torch.device(device_type.value)
    print(f"Using {config.device.value} device")

    model = CheckMatrixModel().to(device)
    if os.path.exists(config.path) and True:
        model.load_state_dict(torch.load(config.path))


    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.SmoothL1Loss()

    get_logger().info("Started")

    running = True
    try:
        epoch = 0
        while running:
            epoch += 1

            start_time = time.time()
            result = play_game(
                model,
                optimizer,
                criterion,
                device,
                config.mcts.iterations,
                config.mcts.workers)
            print(f"Game {epoch}, Result: {result}")
            end_time = time.time()

            get_logger().info(f"Game {epoch}, Result: {result}, Duration: {end_time - start_time}")


            if config.epochs != -1 and epoch >= config.epochs:
                get_logger().info(f"Reached max epochs ({config.epochs}), stopping")
                running = False

    except Exception as e:
        get_logger().error(f"An error occurred: {e}")
        running = False
        raise e
    finally:
        get_logger().info("Stopped")
        torch.save(model.state_dict(), config.path)


if __name__ == "__main__":
    main()
