import sys

import pylightxl as xl
import math
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DEBUG = {
    "fileParser":False,
    "parseData":False,
    "dataSet": False,
    "dataSetSplit": False,
    "getSplitData": False
}

labelDivision = "dependent/independent/control"
#labelDivision = "healthy/unhealthy"

class dataEntry:
    def __init__(self, ID, diagnosis, data, label_division = "dependent/independent/control"):
        self.ID = ID
        self.diagnosis = diagnosis
        if labelDivision == "dependent/independent/control":
            if self.diagnosis == "Healthy control":
                self.label = 0
            if self.diagnosis == "Cushing - ACTH dependent":
                self.label = 1
            if self.diagnosis == "Cushing - ACTH independent":
                self.label = 2
        if labelDivision == "healthy/unhealthy":
            if self.diagnosis == "Healthy control":
                self.label = 0
            if self.diagnosis == "Cushing - ACTH dependent":
                self.label = 1
            if self.diagnosis == "Cushing - ACTH independent":
                self.label = 1
        self.data = np.array(data)
        self.logData = np.array(self.normalize())

    def printData(self):
        string = " | "
        string2 = " | "
        for entry in self.data:
            string = string + str(entry) + " | "
        for entry in self.logData:
            string2 = string2 + str(entry) + " | "
        print("ID: " + self.ID + "\nDiagnosis: " + self.diagnosis + " (" + str(self.label) + ")\nAssociated Data: " + string + "\nLog Data: " + string2)

    def normalize(self):
        out = []
        for entry in self.data:
            out.append(math.log(float(entry)))
        return(out)

class dataSet:
    def __init__(self, data_set, n_splits, test_size = 20, data_type = "log"):
        self.data = []
        self.labels = []
        self.labelNames = []

        for entry in data_set:
            if entry.diagnosis not in self.labelNames:
                self.labelNames.append(entry.diagnosis)

        if DEBUG["dataSet"]:
            print(str(self.labelNames))

        tempSet = data_set

        for entry in tempSet:

            if data_type == "log":
                self.data.append(entry.logData)
            else:
                self.data.append(entry.data)

            self.labels.append(entry.label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        self.splitIndexes = self.test_train_split(n_splits, test_size)

    def test_train_split(self, n_splits, test_size):

        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

        split_result = []
        if DEBUG["dataSetSplit"]:
            print("---SPLITTING DATA---")
        for train_index, test_index in splitter.split(self.data, self.labels):
            split_result.append([train_index, test_index])
            if DEBUG["dataSetSplit"]:
                print("Training Set (size of " + str(len(train_index)) + "):")
                print(train_index)
                print("Testing Set (size of " + str(len(test_index)) + "):")
                print(test_index)

        return(split_result)

    def get_split_data_set(self, split_index):

        train_index, test_index = self.splitIndexes[split_index]

        if DEBUG["getSplitData"]:
            print("Training Set Index " + str(split_index) + ":")
            print("Training Set (size of " + str(len(train_index)) + "):")
            print(train_index)
            print("Testing Set (size of " + str(len(test_index)) + "):")
            print(test_index)

        trainData, testData = self.data[train_index], self.data[test_index]
        trainLabels, testLabels = self.labels[train_index], self.labels[test_index]

        return([trainData, trainLabels], [testData, testLabels])

class fileParser:
    def __init__(self, filePath):
        self.path = filePath
        self.dataNames = []
        self.db = xl.readxl(fn=filePath)
        self.data = self.parseFiles()
        #self.data = self.parseData(self.data)
        if DEBUG["fileParser"]:
            self.printData(self.data)
            print(len(self.data))
        #for entry in self.data:
            #print(len(entry))
            #entry.normalize()

    def parseFiles(self):
        out = []
        counter = 0
        for row in self.db.ws(ws='LC-MS - 24h (mcg24h)').rows:
            if counter != 0:
                outRow = []
                nullEntry = False
                for i in range(5,21):
                    outRow.append(row[i])
                    if row[i] == "":
                        nullEntry = True
                if not nullEntry:
                    newEntry = dataEntry(row[0], row[1], outRow, label_division = labelDivision)
                    out.append(newEntry)
            else:
                for i in range(5,21):
                    self.dataNames.append(row[i])
            counter += 1
        return(out)

    def printData(self, data):
        for entry in data:
            entry.printData()

    def parseData(self, data):
        testArr = []

        if DEBUG["parseData"]:
            print(len(data))

        for entry in data:
            for col in entry.data:
                if col == "":
                    testArr.append(data.index(entry))

        if DEBUG["parseData"]:
            print(len(testArr))
            print(testArr)

        testArr.reverse()

        for entry in testArr:
            data.pop(entry)

        if DEBUG["parseData"]:
            print(len(data))

        return(data)

class processLogger:
    def __init__(self):
        self.states = np.array([])

    def __call__(self, state):
        self.states = np.append(self.states, state)
        return False
