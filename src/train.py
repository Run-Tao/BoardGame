from src.game import NestedTicTacToe
from src.mcts import MCTS
from src.model import TicTacToeNet, NeuralNetworkWrapper
import torch

def self_play(model, num_games):
    mcts = MCTS(model)
    examples = []

    for _ in range(num_games):
        state = NestedTicTacToe()
        states, values = [], []

        while not state.is_terminal():
            action = mcts.search(state)
            states.append(state.clone())
            state.apply_move(action)
            value = state.get_reward()
            values.append(value)

        examples.extend(zip(states, values))

    return examples


def train(model, num_iterations):
    for i in range(num_iterations):
        examples = self_play(model, num_games=100)
        model.train(examples)
        print(f"Iteration {i + 1} complete.")
        print(f"Loss after iteration {i + 1}: {calculate_loss(model, examples)}")
        print(f"Win rate after iteration {i + 1}: {calculate_win_rate(model, num_games=100)}")
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
    for _ in range(num_games):
        state = NestedTicTacToe()
        mcts = MCTS(model)
        while not state.is_terminal():
            action = mcts.search(state)
            state.apply_move(action)
        if state.check_winner() == 'X':
            wins += 1
    return wins / num_games

if __name__ == "__main__":
    model = NeuralNetworkWrapper(TicTacToeNet())
    train(model, num_iterations=10)
