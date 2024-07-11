from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np


class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            state_size = ((2,) * 10 + (4, 4, 4))
            self.q_table = np.zeros(state_size)

        self.lr = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.995
        self.old_distance_from_snack = 0
        self.new_distance_from_snack = 0

    def get_optimal_policy(self, state):
        actions = self.q_table[tuple(state)]
        winner = np.argwhere(actions == np.amax(actions)).flatten().tolist()
        rand = random.randint(0, len(winner) - 1)
        return winner[rand]

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action

    def update_q_table(self, state, action, next_state, reward):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value

    def get_loc_score(self, loc, snack, other_snake):
        score = 1 # empty
        if loc in list(map(lambda z: z.pos, self.body)):
            score = 0 # self body
        if loc in list(map(lambda z: z.pos, other_snake.body)):
            score = 0 # other snake body
        if loc == snack.pos:
            score = 1 # snack
        if loc[0] >= ROWS - 1 or loc[0] < 1 or loc[1] >= ROWS - 1 or loc[1] < 1:
            score = 0 # out of board
        
        return score
    
    def get_around_locs(self):
        locs = [(0, 0)] * 10
        head = self.head.pos
        if self.dirnx == 1: # right
            locs[0] = (head[0] - 1, head[1] - 1)
            locs[1] = (head[0], head[1] - 2)
            locs[2] = (head[0], head[1] - 1)
            locs[3] = (head[0] + 1, head[1] - 1)
            locs[4] = (head[0] + 2, head[1])
            locs[5] = (head[0] + 1, head[1])
            locs[6] = (head[0] + 1, head[1] + 1)
            locs[7] = (head[0], head[1] + 2)
            locs[8] = (head[0], head[1] + 1)
            locs[9] = (head[0] - 1, head[1] + 1)
        elif self.dirny == -1: # up
            locs[0] = (head[0] - 1, head[1] + 1)
            locs[1] = (head[0] - 2, head[1])
            locs[2] = (head[0] - 1, head[1])
            locs[3] = (head[0] - 1, head[1] - 1)
            locs[4] = (head[0], head[1] - 2)
            locs[5] = (head[0], head[1] - 1)
            locs[6] = (head[0] + 1, head[1] - 1)
            locs[7] = (head[0] + 2, head[1])
            locs[8] = (head[0] + 1, head[1])
            locs[9] = (head[0] + 1, head[1] + 1)
        elif self.dirnx == -1: # left
            locs[0] = (head[0] + 1, head[1] + 1)
            locs[1] = (head[0], head[1] + 2)
            locs[2] = (head[0], head[1] + 1)
            locs[3] = (head[0] - 1, head[1] + 1)
            locs[4] = (head[0] - 2, head[1])
            locs[5] = (head[0] - 1, head[1])
            locs[6] = (head[0] - 1, head[1] - 1)
            locs[7] = (head[0], head[1] - 2)
            locs[8] = (head[0], head[1] - 1)
            locs[9] = (head[0] + 1, head[1] - 1)
        elif self.dirny == 1: # down:
            locs[0] = (head[0] + 1, head[1] - 1)
            locs[1] = (head[0] + 2, head[1])
            locs[2] = (head[0] + 1, head[1])
            locs[3] = (head[0] + 1, head[1] + 1)
            locs[4] = (head[0], head[1] + 2)
            locs[5] = (head[0], head[1] + 1)
            locs[6] = (head[0] - 1, head[1] + 1)
            locs[7] = (head[0] - 2, head[1])
            locs[8] = (head[0] - 1, head[1])
            locs[9] = (head[0] - 1, head[1] - 1)

        return locs
    
    def get_relative_loc(self, point):
        loc = 0
        if point.pos[0] <= self.head.pos[0] and point.pos[1] <= self.head.pos[1]:
            loc = 0 # west north
        elif point.pos[0] >= self.head.pos[0] and point.pos[1] <= self.head.pos[1]:
            loc = 1 # east north
        elif point.pos[0] >= self.head.pos[0] and point.pos[1] >= self.head.pos[1]:
            loc = 2 # east south
        elif point.pos[0] <= self.head.pos[0] and point.pos[1] >= self.head.pos[1]:
            loc = 3 # west south

        return loc
    
    def get_state(self, snack, other_snake):
        state = [0] * 12
        locs = self.get_around_locs()
        for i, loc in enumerate(locs):
            state[i] = self.get_loc_score(loc, snack, other_snake)
        state[10] = self.get_relative_loc(snack)

        if self.dirnx == 1: # right
            state[11] = 0
        elif self.dirny == -1: # up
            state[11] = 1
        elif self.dirnx == -1: # left
            state[11] = 2
        elif self.dirny == 1: # down:
            state[11] = 3

        return tuple(state)
    
    def move(self, snack, other_snake):
        state = self.get_state(snack, other_snake)
        action = self.make_action(state)

        self.old_distance_from_snack = abs(self.head.pos[0] - snack.pos[0]) + abs(self.head.pos[1] - snack.pos[1])

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        self.new_distance_from_snack = abs(self.head.pos[0] - snack.pos[0]) + abs(self.head.pos[1] - snack.pos[1]) 
        
        new_state = self.get_state(snack, other_snake)

        return state, new_state, action
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        finished, win_self, win_other = False, False, False
        if self.new_distance_from_snack < self.old_distance_from_snack:
            reward += 2

        if self.check_out_of_board():
            reward += -60
            win_other = True
            finished = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 25
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward += -40
            win_other = True
            reset(self, other_snake)
            finished = True
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                reward += -28
                win_other = True
                finished = True

            else:
                if len(self.body) > len(other_snake.body):
                    reward += 20
                    win_self = True
                    finished = True

                elif len(self.body) == len(other_snake.body):
                    finished = True
                    
                else:
                    reward += -30
                    win_other = True
                    finished = True
                    
            reset(self, other_snake)
            
        return snack, reward, finished, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        