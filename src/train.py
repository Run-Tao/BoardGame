from src.game import NestedTicTacToe
from src.mcts import MCTS
from src.model import TicTacToeNet, NeuralNetworkWrapper
import torch

def self_play(model, num_games):
    mcts = MCTS(model)
    examples = []

    for game_num in range(num_games):
        state = NestedTicTacToe()
        states, values = [], []

        print(f"Starting game {game_num + 1}/{num_games}")
        while not state.is_terminal():
            action = mcts.search(state)
            states.append(state.clone())
            state.apply_move(action)
            value = state.get_reward()
            values.append(value)
            print(f"Game {game_num + 1}: Applied move {action}, reward: {value}")

        examples.extend(zip(states, values))
        print(f"Game {game_num + 1} complete.")

    return examples

def train(model, num_iterations):
    for i in range(num_iterations):
        print(f"Starting iteration {i + 1}/{num_iterations}")
        examples = self_play(model, num_games=100)
        model.train(examples)
        loss = calculate_loss(model, examples)
        win_rate = calculate_win_rate(model, num_games=100)
        print(f"Iteration {i + 1} complete.")
        print(f"Loss after iteration {i + 1}: {loss}")
        print(f"Win rate after iteration {i + 1}: {win_rate}")
        print("---------------------------------------")

def calculate_loss(model, examples):
    total_loss = 0.0
    for state, value in examples:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        value_tensor = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        output = model.predict(state_tensor)
        loss = (output - value_tensor) ** 2
        total_loss += loss.item()
    return total_loss / len(examples)

def calculate_win_rate(model, num_games):
    wins = 0
    for game_num in range(num_games):
        state = NestedTicTacToe()
        mcts = MCTS(model)
        while not state.is_terminal():
            action = mcts.search(state)
            state.apply_move(action)
        if state.check_winner() == 'X':
            wins += 1
        print(f"Evaluated game {game_num + 1}/{num_games}: {'Win' if state.check_winner() == 'X' else 'Loss/Draw'}")
    return wins / num_games

if __name__ == "__main__":
    model = NeuralNetworkWrapper(TicTacToeNet())
    train(model, num_iterations=10)
