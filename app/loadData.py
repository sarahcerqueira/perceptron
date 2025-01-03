from pandas.core.frame import DataFrame
import pandas as pd

class LoadData:

    def getInputOutputFromFile(self, filePath:str, columns:list, inputColumns:list, outputColumn:str ):
        data = self.__loadFile(filePath, columns)
        return self.__separeInputOutput(data, inputColumns, outputColumn)

    def __separeInputOutput(self, data:DataFrame, inputColumns:list, outputColumn:str):
        input = data.filter(items=inputColumns)
        output = data[outputColumn].values.tolist()
        return input,output

    def __loadFile(self, path:str, columns:list):
        return  pd.read_csv(path, sep=' ', header=None, names=columns)