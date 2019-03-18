
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


def playGame(unit):
    gamegrid = GameGrid()

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


        if res != 'notover':
            gamegrid.close()

        if res == 'win' or res == 'lose':
            unit.score = calcScore(res, gamegrid.history_matrixs)
            return (res, unit.score)

def calcScore(result, history_matrixs):
    move_count = len(history_matrixs)
    max_cell = max([max(sub) for sub in history_matrixs[move_count-1]])

    s = move_count + (max_cell*0.7)
    if result == 'win':
        s += 1000
    return s

def createModel(layers):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', input_shape=(16*12,)))
    model.add(Dropout(0.2))
    for layer in layers[1:]:
        model.add(Dense(layer, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))

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
        for i in range(3):
            (result, score) = playGame(unit)
            total_score += score
        unit.score = total_score / 3.0
        print("Unit {} Game Result: {} mean_score: {}".format(a, result, unit.score))

def breed(mum, dad):
    child = copy.copy(mum)
    for j,child_layer in enumerate(child.model.layers):
        #new_weights_for_layer = []

        for k,child_weight_array in enumerate(child_layer.get_weights()):
            dad_weight_array = dad.model.layers[j].get_weights()[k]

            assert(dad_weight_array.shape == child_weight_array.shape)

            save_shape = child_weight_array.shape

            child_weight_array.reshape(-1)
            dad_weight_array.reshape(-1)

            for i,w in enumerate(child_weight_array):
                ratio = random.uniform(0,0.5)
                child_weight_array[i] = child_weight_array[i] * ratio + dad_weight_array[i] * (1-ratio)

            child_weight_array.reshape(save_shape)
            dad_weight_array.reshape(save_shape)
            
            #new_weights_for_layer.append(flattened.reshape(save_shape))
        #child.model.layers[j].set_weights(new_weights_for_layer)

    return child





def optimize():
    population = generateFirstPopulation(20)
    for i in range(10):
        print("Generation: {}".format(i))
        runGeneration(population)

        sortByScore = sorted(population, key=lambda x: x.score, reverse=True)
        
        numToKeep = int(len(sortByScore) * 0.2)
        numChildren = int(len(sortByScore) * 0.8)
        population = np.array(sortByScore[:numToKeep])

        for _ in range(numChildren):
            mum = np.random.choice(population)
            dad = np.random.choice(population)

            child = breed(mum, dad)
            population = np.append(population, [child])
        
        #kill weakest 50%
        #population = np.take(scoredResult, range(0, int(scoredResult.shape[0]*0.5)), axis=1)

optimize()