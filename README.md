# CheckMatrix (Chess AI)

CheckMatrix is a Chess AI that uses Monte Carlo Tree Search (MCTS) along with a transformer-based model to evaluate moves.
It is trained by playing against itself using reinforcement learning.<br>

Currently playing against it is not implemented, though it will be added soon.<br>


## Table of Contents
- [Prerequisites](#prerequisites)
- [Running it](#running-it)

### Prerequisites
- Python 3.10 or above (May work on older versions, but not tested)
- [Stockfish](https://stockfishchess.org/download/) (Optional, but needed if you want it to play against stockfish)

### Running it
1. Clone the repository:
    ```bash
    git clone https://github.com/0x01100101/CheckMatrix.git
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the Virtual Environment, and install the required packages:
    ```bash
    source venv/bin/activate # Linux (Note: I haven't tested this on Linux)
    .\venv\Scripts\activate.bat # Windows
    pip install -r requirements.txt
    ```

4. Run the bot:
    ```bash
    python src/main.py
    ```
