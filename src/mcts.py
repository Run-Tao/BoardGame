import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class MCTS:
    def __init__(self, nnet):
        self.nnet = nnet

    def search(self, state):
        root = Node(state)

        for _ in range(1000):
            node = root
            state_copy = state.clone()

            while node.children:
                node = node.best_child()
                state_copy.apply_move(node.state.last_move)

            if not node.is_fully_expanded():
                action = state_copy.get_random_legal_action()
                state_copy.apply_move(action)
                node = node.add_child(state_copy)

            reward = self.simulate(state_copy)

            while node is not None:
                node.update(reward)
                node = node.parent

        return root.best_child(0).state.last_move

    def simulate(self, state):
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break
            action = state.get_random_legal_action()
            state.apply_move(action)
        return state.get_reward()
