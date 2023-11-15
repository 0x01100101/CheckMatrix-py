import torch
import torch.optim as optim
import os
import time
from logger import logger
from model import CheckMatrixModel
from game import play_game



def main():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_type} device")
    device = torch.device(device_type)

    model = CheckMatrixModel().to(device)
    if os.path.exists("data/model.pth") and True:
        model.load_state_dict(torch.load("data/model.pth"))


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.SmoothL1Loss()

    logger.info("Started")

    running = True
    try:
        epoch = 0
        while running:
            start_time = time.time()
            result = play_game(model, optimizer, criterion, device)
            print(f"Game {epoch+1}, Result: {result}")
            end_time = time.time()

            logger.info(f"Game {epoch+1}, Result: {result}, Duration: {end_time - start_time}")

            epoch += 1
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        running = False
        raise e
    finally:
        logger.info("Stopped")
        torch.save(model.state_dict(), "data/model.pth")


if __name__ == "__main__":
    main()
