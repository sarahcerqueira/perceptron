from random import random
import numpy as np
from pandas.core.frame import DataFrame

class Perceptron:

    def __init__(self):
        self.weights = []

    def train(self,  x_train:DataFrame, y_train:list, learningRate:float):
        self.__startWeights(len(x_train.columns))
        print("Start weight:", self.weights)
        epocas = 0
        isErro = True

        while isErro:
            isErro = False
            for index, row in x_train.iterrows():
                inputList = row.values.tolist()
                yPredicted = self.__predictedY(inputList)

                if yPredicted != y_train[index]:
                    self.__updateWeights(inputList, y_train[index], learningRate, yPredicted)
                    isErro = True

            epocas += 1

        print("End weights:", self.weights)
        print("Epocas:", epocas)

    def __predictedY(self, inputList:list)->float:
        u = self.__weightedSum(inputList)
        return self.__degrau(u)

    def __weightedSum(self, inputList:list)->float:
        weightedSum = 0
        for i in range(len(inputList)):
            weightedSum += inputList[i] * self.weights[i]
        return weightedSum

    def __updateWeights(self, inputList:list, y:float, learningRate:float, yPredicted:float)->None:
        length = len(inputList)
        for i in range(length):
            self.weights[i] = self.weights[i] + learningRate * (y - yPredicted) * inputList[i]

    def __startWeights(self, size:int)->None:
        self.weights = []
        for i in range(size):
            self.weights.append(random())

    def test(self, x_test:DataFrame, y_test:list)->None:
        for index, row in x_test.iterrows():
            predict_class = self.__predictedClass(row.values.tolist())
            print(f'T{index} Class: {predict_class}')

    def __predictedClass(self, inputList):
        yPredicted = self.__predictedY(inputList)
        return self.__getClassName(yPredicted)

    def __getClassName(self, yPredicted:float)->str:
        if yPredicted == -1:
            return "C1"
        else:
            return "C2"

    def __degrau(self, x:float)->float:
        if x>= 0:
            return 1
        return -1