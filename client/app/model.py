import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

class ModelTf:
    def __init__(self):
        self.trainX, self.trainy, self.testX, self.testy = self.load_dataset()
        

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(self.trainX.shape[1],))
        ])

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    
    def trainModel(self, nb_of_parts=1, index=0):
        x_train_part, y_train_part = self.getTrainingData(nb_of_parts, index)
        res = self.model.fit(x_train_part, y_train_part, epochs=1, batch_size=20, verbose=0)
        # print(res)
        return res.history['accuracy'][-1]
    

    def getTrainableVars(self):
        # print(self.model.get_weights()[1].shape)
        return self.model.get_weights()
    

    def toNumpyFlatArray(self):
        flatList = []
        for trainableVar in self.getTrainableVars():
            if len(trainableVar.shape) >= 2:
                # print(trainableVar)
                for item in trainableVar.flatten().tolist():
                    flatList.append(item)
            else:
                for item in trainableVar.tolist():
                    flatList.append(item)
        flatArray = np.asarray(flatList, dtype=np.dtype('d'))
        # print(flatArray)
        return flatArray

    def updateFromNumpyFlatArray(self, flatArray):
        weights = np.array([[np.copy(x)] for x in flatArray[:flatArray.shape[0]-1]])
        bias = np.array([np.copy(flatArray[-1])])
        new_weights = [weights, bias]
        self.model.set_weights(new_weights)

    def getTrainingData(self, nb_of_parts=1, index=0):
        x_start_index = round(index * self.trainX.shape[0]/nb_of_parts)
        y_start_index = round(index * self.trainy.shape[0]/nb_of_parts)

        x_stop_index = x_start_index + round(self.trainX.shape[0]/nb_of_parts)
        y_stop_index = y_start_index + round(self.trainy.shape[0]/nb_of_parts)
        x_stop_index = min(x_stop_index, self.trainX.shape[0])
        y_stop_index = min(y_stop_index, self.trainy.shape[0])

        x_train_part = self.trainX[x_start_index : x_stop_index]
        y_train_part = self.trainy[y_start_index : y_stop_index]
        print(x_train_part.shape, y_train_part.shape)

        return x_train_part, y_train_part
    

    def load_dataset(self, prefix='app'):
        # load all train
        train = pd.read_csv('app/bank/train.csv')
        test = pd.read_csv('app/bank/test.csv')

        trainX = train.drop('y', axis=1) 
        trainy = train['y']  

        testX = test.drop('y', axis=1) 
        testy = test['y']

        scaler = StandardScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)

        return trainX, trainy, testX, testy

    # def train_model(self, epochs=100, batch_size=10):
    #     # Train the model
    #     self.model.fit(self.X_train_scaled, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    def evaluate_model(self):
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(self.testX, self.testy)
        print(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')
        return test_accuracy, test_loss


if __name__ == '__main__':
    MODEL = ModelTf()
    acc = []
    loss = []

    for i in range(50):
        MODEL.trainModel(5, 0)
        test_acc, test_loss = MODEL.evaluate_model()
        acc.append(test_acc)
        loss.append(test_loss)

        weights = MODEL.toNumpyFlatArray().copy()
        filename = 'app/updates/00.csv'
        df = pd.DataFrame([weights])
        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
        
        filename = 'app/updates/accuracy_1.csv'
        dfacc = pd.DataFrame([test_acc])
        file_exists = os.path.isfile(filename)
        dfacc.to_csv(filename, mode='a', index=False, header=not file_exists)

        filename = 'app/updates/loss_1.csv'
        dfloss = pd.DataFrame([test_loss])
        file_exists = os.path.isfile(filename)
        dfloss.to_csv(filename, mode='a', index=False, header=not file_exists)
    
    # weights = MODEL.toNumpyFlatArray()
    print(acc)
    print(loss)
    # MODEL.updateFromNumpyFlatArray(weights)
