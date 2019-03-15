
from puzzle import GameGrid
import logic
import random
import time

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout


def playGame(model):
    gamegrid = GameGrid()

    moves = ['up', 'down', 'left', 'right']

    state = 'not over'
    while state == 'not over':

        flattened = np.array([np.array(gamegrid.matrix).flatten()])
        result = model.predict(x=flattened)[0]
        indices = np.argsort(result)

        res = 'nomove'
        while res == 'nomove':
            if len(indices) == 0:
                return ('stuck', len(gamegrid.history_matrixs))

            nextMove = moves[indices[0]]
            (matrix, res) = gamegrid.makeMove(nextMove)
            if res == 'nomove':
                indices = np.delete(indices, 0)


        if res != 'notover':
            gamegrid.close()

        if res == 'win' or res == 'lose':
            return (res, len(gamegrid.history_matrixs))

def createModel(layers):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', input_shape=(16,)))
    model.add(Dropout(0.2))
    for layer in layers[1:]:
        model.add(Dense(layer, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.2))

    model.add(Dense(4, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))

    return model

def generateFirstPopulation(count):
    models = np.array([])
    for i in range(count):
        layers = np.array([])
        layerCount = random.randint(1, 5)
        for layer in range(1,layerCount+1):
            layers = np.append(layers, [random.randint(64, 512)])
        print(layers)
        model = createModel(layers.astype(int))
        models = np.append(models, [model])
    return models

def runGeneration(models):
    results = np.array([])
    for model in models:
        (result, moveCount) = playGame(model)
        print("Game Result: {} movecount: {}".format(result, moveCount))
        score = calcScore(result, moveCount)
        np.append(results, [(score, model)])

    return results

def calcScore(result, moveCount):
    s = moveCount
    if result == 'win':
        s += 1000
    return s


def optimize():
    population = generateFirstPopulation(50)
    for i in range(6):
        scoredResult = runGeneration(population)
        #scoredResult = np.sort(scoredResult, axis=0)
        scoredResult.sort(key=lambda x: x[0])
        #kill weakest 50%
        scoredResult = np.take(scoredResult, [0,scoredResult.shape[0] * 0.5])

optimize()