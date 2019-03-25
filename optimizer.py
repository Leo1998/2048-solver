
from puzzle import GameGrid
import logic
import random
import time
import copy
import math

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

class Unit(object):
    score = 0
    model = None

    def __str__(object): return 'Unit({})'.format(score)

def cellToOneHot(cell):
    if (cell > 0):
        return int(math.log(cell, 2))
    else:
        return 0


gamegrid = GameGrid()

def playGame(unit):
    gamegrid.init_matrix()
    gamegrid.update_grid_cells()

    moves = ['up', 'down', 'left', 'right']

    state = 'not over'
    while state == 'not over':
#flattened = np.array([np.array(gamegrid.matrix).flatten()])
        onehots = np.zeros((16, 12))
        for i,cell in enumerate(np.array(gamegrid.matrix).flatten()):
            onehots[i][cellToOneHot(cell)] = 1.0
        
        flattened = np.expand_dims(onehots.flatten(), axis=0)

        result = unit.model.predict(x=flattened)[0]
        indices = np.argsort(result)

        res = 'nomove'
        while res == 'nomove':
            if len(indices) == 0:
                return ('stuck', -1)

            nextMove = moves[indices[0]]
            (matrix, res) = gamegrid.makeMove(nextMove)
            if res == 'nomove':
                indices = np.delete(indices, 0)

        if res == 'win' or res == 'lose':
            unit.score = calcScore(res, gamegrid.history_matrixs)
            return (res, unit.score)

def calcScore(result, history_matrixs):
    move_count = len(history_matrixs)
    max_cell = max([max(sub) for sub in history_matrixs[move_count-1]])

    s = max_cell
    if result == 'win':
        s += 1000
    return s

def createModel(layers):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', input_shape=(16*12,)))
    model.add(Dropout(rate=0.9))
    for layer in layers[1:]:
        model.add(Dense(layer, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(rate=0.9))

    model.add(Dense(4, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))

    return model

def generateFirstPopulation(count):
    print("Generating {} units in the first Generation".format(count))
    units = np.array([])
    for i in range(count):
        layers = np.array([96, 64])
        """layers = np.array([])
        layerCount = random.randint(1, 5)
        for layer in range(1,layerCount+1):
            layers = np.append(layers, [random.randint(64, 512)])"""
        #print(layers)
        model = createModel(layers.astype(int))

        unit = Unit()
        unit.model = model

        units = np.append(units, [unit])
    return units

def runGeneration(units):
    for a,unit in enumerate(units):
        total_score = 0
        for i in range(2):
            (result, score) = playGame(unit)
            total_score += score
        unit.score = total_score / 2.0
        print("Unit {} Game Result: {} mean_score: {}".format(a, result, unit.score))

def breed(mum, dad):
    offspring = copy.copy(mum)
    crossover_ratio = random.random()
    for j,offspring_layer in enumerate(offspring.model.layers):
        #new_weights_for_layer = []

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
    
    return offspring

def mutate(unit, mutation_chance):
    for j,layer in enumerate(unit.model.layers):
        for k,weight_array in enumerate(layer.get_weights()):
            save_shape = weight_array.shape

            weight_array.reshape(-1)
            
            for i,w in enumerate(weight_array):
                if (random.random() < mutation_chance):
                    weight_array[i] += random.random() * 6.0 - 3.0
            
            weight_array.reshape(save_shape)
    
    return unit

def optimize():
    population = generateFirstPopulation(20)
    for i in range(10):
        print("Generation: {}".format(i))
        runGeneration(population)

        sortByScore = sorted(population, key=lambda x: x.score, reverse=True)
        
        numToKeep = int(len(sortByScore) * 0.1)
        numOffspring = int(len(sortByScore) * 0.9)

        #only keep the best
        population = np.array(sortByScore[:numToKeep])

        #produce offspring
        for _ in range(numOffspring):
            mum = np.random.choice(population)
            dad = np.random.choice(population)

            offspring = breed(mum, dad)
            mutate(offspring, 0.1)

            population = np.append(population, [offspring])

optimize()
gamegrid.close()