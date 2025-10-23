from copy import deepcopy
from typing import Optional
import random
import math
import os

grid_size = 3

class GameState:
    def __init__(self, state: Optional[list[list[int]]] = None, grid_size: int = grid_size):
        self.grid_size = grid_size
        self.state = state
        self.idx_to_label = {0: "   ", 1: " X ", 2: " O "}
        if self.state is None:
            self.state = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        assert len(self.state) == self.grid_size

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        header = f"Grid size: {self.grid_size}\n"
        lines = [
            "|".join(map(lambda x: self.idx_to_label[x], row))
            for row in self.state
        ]
        return header +("\n"+"--- "*self.grid_size+"\n").join(lines)
    

class GameNode:
    def __init__(self, state: GameState = GameState(), current_player: int = 1, parent: Optional['GameNode'] = None):
        self.value: int = 0
        self.num_visits: int = 0
        self.parent: Optional['GameNode'] = parent
        
        self.state = state
        self.children: list[GameNode] = []
        self.current_player: int = current_player
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.valid_moves: list[tuple[int, int]] = []

        self.game_over: bool = self.game_is_over()

    def ucb(self, c: float = 1.0) -> float:
        if self.num_visits == 0:
            return float('inf')
        return self.value / (self.num_visits) + c * math.sqrt(math.log(self.parent.num_visits) / (self.num_visits))
    
    def _select_next_move(self) -> 'GameNode':
        if not self.children:
            self.generate_children()

        unexplored_children = [child for child in self.children if child.num_visits == 0]
        if unexplored_children:
            return random.choice(unexplored_children)
        
        if self.game_over:
            return self

        current_values = [child.ucb() for child in self.children]
        max_value_indices = [i for i, v in enumerate(current_values) if v == max(current_values)] # get indices of max values
        next_move_idx = random.choice(max_value_indices)
        next_move = self.children[next_move_idx]
        return next_move._select_next_move()
    
    def _expand(self) -> None:
        self.generate_children()

    def _random_move(self) -> 'GameNode':
        next_move = random.choice(self.valid_moves)
        new_state = deepcopy(self.state.state)
        new_state[next_move[0]][next_move[1]] = self.current_player
        return GameNode(
            state=GameState(new_state), 
            current_player=self.current_player % 2 + 1, 
            parent=self)

    def _simulate(self) -> int:
        current_node = self
        while not current_node.game_over:
            current_node = current_node._random_move()

        if current_node.winner == 1: # computer
            return 1
        elif current_node.winner == 2: # human
            return -1
        else:
            return 0 # draw
        
    def _backpropagate(self, value: int) -> None:
        current_node = self
        while current_node:
            current_node.num_visits += 1
            current_node.value += value
            current_node = current_node.parent
    
    def best_next_move(self) -> 'GameNode':
        current_values = [child.ucb(c=0) for child in self.children]
        max_value_indices = [i for i, v in enumerate(current_values) if v == max(current_values)] # get indices of max values
        next_move_idx = random.choice(max_value_indices)
        next_move = self.children[next_move_idx]
        return next_move
    
    def mcts_step(self) -> None:
        next_move = self._select_next_move()
        if next_move.game_over:
            next_move._backpropagate(1 if next_move.winner == 1 else 0)
        else:
            next_move._expand()
            value = next_move._simulate()
            next_move._backpropagate(value)

    def mcts_loop(self, iterations: int) -> None:
        for _ in range(iterations):
            self.mcts_step()

    def add_child(self, child: 'GameNode') -> None:
        self.children.append(child)

    def remove_child(self, child: 'GameNode') -> None:
        self.children.remove(child)

    def generate_children(self, recursive: bool = False) -> None:
        if self.children:
            return
        
        if self.game_over:
            return
        
        for i, j in self.valid_moves:
            new_state = deepcopy(self.state.state)
            new_state[i][j] = self.current_player
            new_child = GameNode(
                state=GameState(new_state), 
                current_player=self.current_player % 2 + 1, 
                parent=self)
            self.add_child(new_child)

            if recursive:
                new_child.generate_all_children(recursive=recursive)

    def get_valid_moves(self) -> list[tuple[int, int]]:
        valid_moves = []
        if self.game_over:
            return valid_moves
        
        for i in range(self.state.grid_size):
            for j in range(self.state.grid_size):
                if self.state.state[i][j] == 0:
                    valid_moves.append((i, j))
        return valid_moves

    def game_is_over(self) -> bool:
        game_over = False
        last_player = self.current_player % 2 + 1
        for i in range(self.state.grid_size):
            game_over += all([v == last_player for v in self.state.state[i]]) # check ith line
            game_over += all([self.state.state[j][i] == last_player for j in range(self.state.grid_size)]) # check ith col

        # check diagonals
        game_over += all([self.state.state[i][i] == last_player for i in range(self.state.grid_size)]) # check main diagonal
        game_over += all([self.state.state[i][self.state.grid_size - 1 - i] == last_player for i in range(self.state.grid_size)])

        if game_over:
            self.winner = last_player
        
        self.valid_moves = self.get_valid_moves()
        if not self.valid_moves:
            game_over = True

        return bool(game_over)
    
    def print_tree(self, depth: int = 0) -> None:
        if self.num_visits > 0:
            print(str(self))
            for child in self.children:
                child.print_tree(depth=depth+1)
    
    def __repr__(self):
        return f"{self.state} \nvalue={self.value}, num_visits={self.num_visits}, current_player={self.current_player}"


if __name__ == "__main__":
    # Game loop
    root = GameNode(
        state=GameState(),
    )
    current_node = root
    while not current_node.game_over:
        # Computer turn
        current_node.mcts_loop(1000) 
        computer_move = current_node.best_next_move()
        current_node = computer_move
        os.system('cls')
        print("Computer move:\n", current_node)
        print("\n"+ "---"*20 + "\n")

        if current_node.value <= 0:
            print("Computer calls for a draw.")
            answer = input("Accept (y/n)? ")
            if answer == "y":
                current_node.winner = 0
                current_node.game_over = True
        
        if current_node.game_over:
            break

        # Human turn --- improved move listing and input validation
        current_node.generate_children()

        print("Available moves:")
        for idx, child in enumerate(current_node.children):
            move = current_node.valid_moves[idx]
            print(f"{idx}: move={move}")
            # show board for this child (skip the first header line from GameState.__repr__)
            board_lines = str(child.state).splitlines()
            for line in board_lines[1:]:
                print("    " + line)
            print("-" * 20)

        # validated input loop
        while True:
            s = input(f"Enter move index (0-{len(current_node.children)-1}): ")
            try:
                if s == "q":
                    break
                human_move = int(s)
                if 0 <= human_move < len(current_node.children):
                    break
            except ValueError:
                pass
            print("Invalid selection, try again.")

        os.system('cls')
        current_node = current_node.children[human_move]
        print("Human move:\n", current_node, flush=True)
        print("\n" + "---" * 20 + "\n")
    
    os.system('cls')
    print('Final board:\n', current_node)
    winner = None
    if current_node.winner == 1: winner = "Computer"
    elif current_node.winner == 2: winner = "Human"
    else: winner = "Draw"
    print('Winner:', winner)