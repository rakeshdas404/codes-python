import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyQt5 import QtWidgets
import argparse

class Nonogram:
    def __init__(self, row_clues, col_clues):
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.num_rows = len(row_clues)
        self.num_cols = len(col_clues)
        self.grid = np.full((self.num_rows, self.num_cols), -1)  # -1 for unknown

    def is_consistent(self, line, clues):
        filled_segments = [len(list(group)) for key, group in itertools.groupby(line) if key == 1]
        if len(filled_segments) != len(clues):
            return False
        return all(filled_segments[i] == clues[i] for i in range(len(clues)))

    def propagate_constraints(self):
        while True:
            changed = False
            for i in range(self.num_rows):
                valid_lines = self.find_valid_lines(self.grid[i, :], self.row_clues[i])
                new_line = self.intersect_lines(valid_lines)
                if not np.array_equal(self.grid[i, :], new_line):
                    self.grid[i, :] = new_line
                    changed = True
            for j in range(self.num_cols):
                valid_lines = self.find_valid_lines(self.grid[:, j], self.col_clues[j])
                new_line = self.intersect_lines(valid_lines)
                if not np.array_equal(self.grid[:, j], new_line):
                    self.grid[:, j] = new_line
                    changed = True
            if not changed:
                break

    def find_valid_lines(self, line, clues):
        possible_lines = []
        for perm in itertools.product([0, 1], repeat=len(line)):
            if self.is_consistent(perm, clues) and all(line[k] == -1 or line[k] == perm[k] for k in range(len(line))):
                possible_lines.append(perm)
        return possible_lines

    def intersect_lines(self, valid_lines):
        return [self.common_value(positions) for positions in zip(*valid_lines)]

    def common_value(self, values):
        if all(value == values[0] for value in values):
            return values[0]
        return -1

    def solve(self):
        self.propagate_constraints()
        return self.grid

    def display(self):
        app = QtWidgets.QApplication([])

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Drawing the grid
        for y in range(self.num_rows):
            for x in range(self.num_cols):
                color = 'black' if self.grid[y, x] == 1 else 'white'
                ax.add_patch(Rectangle((x, y), 1, 1, edgecolor='black', facecolor=color))

        # Adding clues
        for y, clue in enumerate(self.row_clues):
            ax.text(-1, y + 0.5, ' '.join(map(str, clue)), va='center', ha='right', fontsize=10)
        for x, clue in enumerate(self.col_clues):
            ax.text(x + 0.5, -1, '\n'.join(map(str, clue)), va='top', ha='center', fontsize=10, rotation=0)

        ax.set_xlim(-2, self.num_cols)
        ax.set_ylim(-2, self.num_rows)
        ax.invert_yaxis()
        plt.axis('off')

        fig.show()
        # Saving the plot as an image file
        #plt.savefig(f"{save_path}/solved image {image_name}.png")
        #plt.show()

        app.exec_()

# Example usage
def parse_clues(file_path):
    """Parse clues from file and return row and column clues."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    row_clues = []
    column_clues = []
    is_row = True  # Toggle between row and column clues
    for line in lines:
        line = line.strip()
        if line == "":
            is_row = False
            continue
        clues = list(map(int, line.split()))
        if is_row:
            row_clues.append(clues)
        else:
            column_clues.append(clues)
    return row_clues, column_clues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a nonogram from a file conatining row and column clues") 
    parser.add_argument("file_path", type=str, help="Path to the clue file")     
    args = parser.parse_args()
    file_path = args.file_path  # Path to your clues file
    #save_path= os.getcwd()
    #image_name=file_path[-1]
    row_clues, col_clues = parse_clues(file_path)
    nonogram = Nonogram(row_clues, col_clues)
    solution = nonogram.solve()
    nonogram.display()
    

