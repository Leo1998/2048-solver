
from puzzle import *
import logic
import random
import time
import copy
import math

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import RandomUniform

class Unit(object):
    score = 0
    max_cell = 0
    model = None

    def __str__(self): return 'Unit(score: {}, max_cell: {})'.format(score, max_cell)


def createModel(layers):
    initializer = RandomUniform(minval=-1.0, maxval=+1.0, seed=None)

    model = Sequential()
    model.add(Dense(layers[0], activation='relu', kernel_initializer=initializer, bias_initializer=initializer, input_shape=(16,)))
    #model.add(Dropout(rate=0.9))
    for layer in layers[1:]:
        model.add(Dense(layer, activation='relu', kernel_initializer=initializer, bias_initializer=initializer))
        #model.add(Dropout(rate=0.9))

    model.add(Dense(4, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer))

    return model

def generateFirstPopulation(count):
    print("Generating {} units in the first Generation".format(count))
    units = np.array([])
    for i in range(count):
        layers = np.array([16])
        #print(layers)
        model = createModel(layers.astype(int))

        unit = Unit()
        unit.model = model

        units = np.append(units, [unit])
    return units

def calcScore(result, history_matrixs, max_cell, illegal_moves_tried):
    move_count = len(history_matrixs)

    s = (move_count - illegal_moves_tried)
    if result == 'win':
        s += 10000000
    return s

def breed(mum, dad, crossover_ratio):
    offspring = copy.copy(mum)
    
    for j,offspring_layer in enumerate(offspring.model.layers):
        new_weights_for_layer = []

        for k,offspring_weight_array in enumerate(offspring_layer.get_weights()):
            dad_weight_array = dad.model.layers[j].get_weights()[k]

            assert(dad_weight_array.shape == offspring_weight_array.shape)

            save_shape = offspring_weight_array.shape

            offspring_weight_array.reshape(-1)
            dad_weight_array.reshape(-1)

            for i,w in enumerate(offspring_weight_array):
                offspring_weight_array[i] = offspring_weight_array[i] * crossover_ratio + dad_weight_array[i] * (1-crossover_ratio)

            offspring_weight_array.reshape(save_shape)
            dad_weight_array.reshape(save_shape)

            new_weights_for_layer.append(offspring_weight_array)

        offspring_layer.set_weights(new_weights_for_layer)
    
    return offspring

def mutate(unit, mutation_chance):
    for j,layer in enumerate(unit.model.layers):
        new_weights_for_layer = []

        for k,weight_array in enumerate(layer.get_weights()):
            save_shape = weight_array.shape

            weight_array.reshape(-1)
            
            for i,w in enumerate(weight_array):
                if (random.random() < mutation_chance):
                    weight_array[i] += random.uniform(-1, +1)
            
            weight_array.reshape(save_shape)
            new_weights_for_layer.append(weight_array)

        layer.set_weights(new_weights_for_layer)
    
    return unit







gamegrid = GameGrid()

def playGame(unit, headless=False):
    game = Game()

    if not headless:
        gamegrid.update_grid_cells(game.matrix)

    moves = ['up', 'down', 'left', 'right']

    illegal_moves_tried = 0

    state = 'not over'
    while state == 'not over':
        data = np.zeros((4, 4))
        normalize_factor = np.max(game.matrix)
        for i in range(4):
            for j in range(4):
                data[i][j] = game.matrix[i][j] / normalize_factor 
        #print(data)
        
        flattened = np.expand_dims(data.flatten(), axis=0)

        result = unit.model.predict(x=flattened)[0]
        indices = np.argsort(result)[:2]

        res = 'nomove'
        while res == 'nomove':
            nextMove = moves[indices[0]]
            (matrix, res) = game.makeMove(nextMove)
            if res == 'nomove':
                illegal_moves_tried += 1
                indices = np.delete(indices, 0)
            if len(indices) == 0:
                res = 'stuck'

        if not headless:
            gamegrid.update_grid_cells(game.matrix)

        if res == 'win' or res == 'lose' or res == 'stuck':
            max_cell = max([max(sub) for sub in game.history_matrixs[len(game.history_matrixs)-1]])
            unit.score = calcScore(res, game.history_matrixs, max_cell, illegal_moves_tried)
            unit.max_cell = max(unit.max_cell, max_cell)
            return (res, unit.score)

def runGeneration(units, num_games):
    for a,unit in enumerate(units):
        unit.max_cell = 0
        total_score = 0
        for _ in range(num_games):
            (result, score) = playGame(unit)
            total_score += score
        unit.score = total_score / num_games
        print("Unit {} Game Result: {} mean_score: {}, max_cell:{}".format(a, result, unit.score, unit.max_cell))


def optimize():
    population = generateFirstPopulation(20)
    for i in range(60):
        print("Generation: {}".format(i))
        runGeneration(population, 20)

        sortByScore = sorted(population, key=lambda x: x.score, reverse=True)
        
        keep_rate = 0.4
        numToKeep = int(len(sortByScore) * keep_rate)
        numOffspring = int(len(population) * (1 - keep_rate))

        #only keep the best
        population = np.array(sortByScore[:numToKeep])

        #produce offspring
        for _ in range(numOffspring):
            mum = np.random.choice(population)
            dad = np.random.choice(np.delete(population, population.tolist().index(mum)))
            assert(mum != dad)

            crossover_ratio = random.random()
            
            offspring = breed(mum, dad, crossover_ratio)

            # mutation chance is random in the bounds of [almost healthy, full retard)
            mutation_chance = random.uniform(0.005, 0.3)

            mutate(offspring, mutation_chance)

            population = np.append(population, [offspring])

optimize()
gamegrid.close()
