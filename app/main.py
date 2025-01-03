import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from perceptron import Perceptron
from app.config import *
from app.loadData import LoadData

class Main:

    def run(self):
        loadData = LoadData()
        x_train, y_train = loadData.getInputOutputFromFile(TRAIN_FILE_PATH, COLUMNS, INPUT_COLUMNS, OUTPUT_COLUMN)
        x_test, y_test = loadData.getInputOutputFromFile(TEST_FILE_PATH, COLUMNS, INPUT_COLUMNS, OUTPUT_COLUMN)

        perceptron = Perceptron()
        perceptron.train(x_train, y_train, LEARNING_RATE)
        perceptron.test(x_test, y_test)
        print(type(x_train))

main= Main()
main.run()
