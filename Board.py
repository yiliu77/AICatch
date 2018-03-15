import pygame
import time
import random
import numpy as np
from AICatch.NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt

class Status:
    IN_PROGRESS = 0
    GAME_OVER = 1


class AI:
    CAUGHT = 0
    MISSED = 1
    IN_PROGRESS = 2
    HIT_SIDE = 3


class Board:
    def __init__(self, side_length, width, length, block_refresh_rate, ai_refresh_rate, until_next_block, ai_trained):
        # side_length is how wide each square is
        self.side_length = side_length
        # number of squares of length of board
        self.length = length
        # number of squares of width of board
        self.width = width

        # refresh blocks coming down rate
        self.block_refresh_rate = block_refresh_rate
        # refresh ai decision rate
        self.ai_refresh_rate = ai_refresh_rate

        self.until_block_increment = 0
        # until next block
        self.until_next_block = until_next_block
        if until_next_block <= 0:
            raise Exception('Game has to have space between blocks for AI to work')

        self.ai_status = AI.IN_PROGRESS
        # all blocks in game
        self.blocks = []
        # catcher
        center = int(width * (length - 1) + length / 2)
        self.catcher = [center - 1, center, center + 1]

        # timers
        self.block_timer = time.time()
        self.ai_timer = time.time()

        self.score = 0
        self.misses = 0
        self.all_percentages = []

        # ai
        input_nodes = int(width * length)
        hidden_nodes = 300
        output_nodes = 3
        learning_rate = 0.001
        if ai_trained:
            ai_catch_who = open("ai_catch_who_50_88.37209302325581.csv", 'r')
            ai_who_read = ai_catch_who.readlines()
            ai_catch_who.close()
            ai_catch_wih = open("ai_catch_wih_50_88.37209302325581.csv", 'r')
            ai_wih_read = ai_catch_wih.readlines()
            ai_catch_wih.close()

            weight_wih = np.asfarray([line.split(',') for line in ai_wih_read])
            weight_who = np.asfarray([line.split(',') for line in ai_who_read])
        else:
            weight_wih = np.random.randn(hidden_nodes, int(input_nodes)) / np.sqrt(input_nodes)
            weight_who = np.random.randn(output_nodes, hidden_nodes) / np.sqrt(hidden_nodes)

        self.ai = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, weight_wih, weight_who,
                                learning_rate)

    # turns board into 2D matrix representation so AI can understand it (not convolutional network :P)
    def representation(self):
        board_rep = []
        for i in range(self.length * self.width):
            if i in self.blocks:
                board_rep.append(0.5)
            elif i in self.catcher:
                board_rep.append(0.98)
            else:
                board_rep.append(0.02)
        return board_rep

    def check_game_status(self):
        for square in self.catcher:
            if square in self.blocks:
                return Status.GAME_OVER
        return Status.IN_PROGRESS

    def update_blocks(self):
        if time.time() - self.block_timer > self.block_refresh_rate:
            if self.until_block_increment >= self.until_next_block:
                self.blocks.append(random.randint(0, self.width - 1))
                self.until_block_increment = 0
            else:
                self.until_block_increment += 1

            self.blocks = [x + self.width for x in self.blocks]
            for block in self.blocks:
                # remove block if outside board
                if block >= self.length * self.width:
                    self.blocks.remove(block)
                    self.ai_status = AI.MISSED
                elif block in self.catcher:
                    self.blocks.remove(block)
                    self.ai_status = AI.CAUGHT
            # restart block timer
            self.block_timer = time.time()

    def get_ai_status(self):
        return self.ai_status

    def reset_ai_status(self):
        self.ai_status = AI.IN_PROGRESS

    def reset_scores_misses(self):
        self.score = 0
        self.misses = 0

    def update_catcher(self):
        if time.time() - self.ai_timer > self.ai_refresh_rate:
            # prevents program from crashing for some reason
            pygame.event.get()

            move = np.argmax(self.ai.query(self.representation()))
            if move == 0:
                if not self.catcher[0] <= self.width * (self.length - 1):
                    self.catcher = [i - 1 for i in self.catcher]
                else:
                    self.ai_status = AI.HIT_SIDE
            if move == 2:
                if not self.catcher[-1] >= self.width * self.length - 1:
                    self.catcher = [i + 1 for i in self.catcher]
                else:
                    self.ai_status = AI.HIT_SIDE

            for block in self.blocks:
                if block in self.catcher:
                    self.ai_status = AI.CAUGHT
                    self.blocks.remove(block)

            self.ai_timer = time.time()

    # draw square based on position of block
    def square_equation(self, position):
        return (
            (position % self.width) * self.side_length, int(position / self.width) * self.side_length, self.side_length,
            self.side_length)

    def init_draw(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.side_length * self.width, self.side_length * self.length))

        self.black = (0, 0, 0, 255)
        self.white = (255, 255, 255)
        self.gray = (220, 220, 220)
        self.silver = (192, 192, 192)
        self.red = (255, 0, 0)

    def display_board(self):
        self.window.fill(self.white)

        for catcher in self.catcher:
            pygame.draw.rect(self.window, self.red, self.square_equation(catcher))
        for block in self.blocks:
            pygame.draw.rect(self.window, self.black, self.square_equation(block))

        pygame.display.update()

    def player_play(self):
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_LEFT:
                    self.catcher = [i - 1 for i in self.catcher]

                if e.key == pygame.K_RIGHT:
                    self.catcher = [i + 1 for i in self.catcher]

            if e.type == pygame.QUIT:
                pygame.quit()
                return True

        pygame.display.update()
        return False

    def reset_ai(self):
        self.ai.reset()

    def train_ai(self, caughtIt):
        self.ai.train(caughtIt)

    def increase_score(self):
        self.score += 1

    def get_score(self):
        return self.score

    def increase_misses(self):
        self.misses += 1

    def get_misses(self):
        return self.misses

    def get_percentages(self):
        return self.all_percentages

    def add_percentages(self):
        self.all_percentages.append(float(self.score * 100) / float(self.score + self.misses))

    def save_weights(self, increment):
        self.ai.save(str(increment) + '_' + str(self.all_percentages[-1]))

if __name__ == "__main__":
    board = Board(70, 9, 9, 0.060, 0.03, 2, True)
    increment = 35

    board.init_draw()
    while True:
        board.update_blocks()
        board.update_catcher()
        board.display_board()

        if board.get_ai_status() == AI.CAUGHT:
            board.increase_score()
            board.train_ai(True)
            board.reset_ai()
            print("============================")
            print("--------------------")
            if board.get_misses() > 0:
                print(str(board.get_score()) + " , " + str(board.get_misses()) + " " + str(
                    float(100 * board.get_score()) / (board.get_score() + float(board.get_misses()))) + "%")
            print("--------------------")

        if board.get_ai_status() == AI.MISSED:
            board.increase_misses()
            board.train_ai(False)
            board.reset_ai()
            print("============================")
            print("--------------------")
            if board.get_misses() > 0:
                print(str(board.get_score()) + " , " + str(board.get_misses()) + " " + str(
                    float(100 * board.get_score()) / (board.get_score() + float(board.get_misses()))) + "%")
            print("--------------------")

        if board.get_ai_status() == AI.HIT_SIDE:
            board.train_ai(False)
            board.reset_ai()

        if board.get_misses() + board.get_score() > 300:
            board.add_percentages()
            board.reset_scores_misses()
            # only save weights and plot every 10 times
            if len(board.get_percentages()) % 7 == 0:
                plt.scatter([i for i in range(len(board.get_percentages()))], board.get_percentages(), color="red")
                board.save_weights(increment)
                plt.show()
                increment += 1

        board.reset_ai_status()
