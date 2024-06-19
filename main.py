# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# class import
from src.game import UltimateTicTacToe
from src.model import TicTacToeNet
from src.mcts import MCTSNode

# python library
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
import torch.nn as nn


def train():
    game = UltimateTicTacToe()
    model = TicTacToeNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_value = nn.MSELoss()
    criterion_policy = nn.CrossEntropyLoss()

    for episode in range(1000):  # number of episodes
        state = game.reset()
        states = []
        actions = []
        rewards = []

        # Self-play game
        while True:
            root = MCTSNode(state)
            for _ in range(1600):  # number of simulations
                leaf = root.select()
                reward = simulate_game(leaf.state)
                leaf.expand()
                leaf.backpropagate(reward)

            best_move = root.best_child(c_param=0).state.last_move
            states.append(state)
            actions.append(best_move)
            state, main_board, done = game.step(best_move)
            rewards.append(game.check_winner(main_board))
            if done:
                break

        # Convert states and actions to training data
        state_tensors = torch.tensor([state_to_tensor(s) for s in states], dtype=torch.float32)
        action_tensors = torch.tensor([action_to_index(a) for a in actions], dtype=torch.long)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32)

        # Train model
        optimizer.zero_grad()
        values, policies = model(state_tensors)
        loss_value = criterion_value(values, reward_tensors)
        loss_policy = criterion_policy(policies, action_tensors)
        loss = loss_value + loss_policy
        loss.backward()
        optimizer.step()


def simulate_game(state):
    game = UltimateTicTacToe()
    game.board = np.copy(state.board)
    game.main_board = np.copy(state.main_board)
    game.current_player = state.current_player
    game.last_move = state.last_move
    while not game.is_full(game.main_board):
        legal_moves = game.get_legal_moves()
        move = legal_moves[np.random.choice(len(legal_moves))]
        _, main_board, done = game.step(move)
        if done:
            break
    return game.check_winner(game.main_board)


def state_to_tensor(state):
    tensor = np.zeros((2, 9, 9), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            for x in range(3):
                for y in range(3):
                    if state.board[i, j, x, y] == 1:
                        tensor[0, i * 3 + x, j * 3 + y] = 1
                    elif state.board[i, j, x, y] == -1:
                        tensor[1, i * 3 + x, j * 3 + y] = 1
    return tensor


def action_to_index(action):
    i, j, x, y = action
    return i * 27 + j * 9 + x * 3 + y


train()
# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
