import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow
import matplotlib.pyplot as plt
import os
import gc
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
# importing keras layers for cnn
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model, load_model
# splitting the data to train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from exception  import CustomException
from logger import logging

# File locations
files = os.listdir('data')
# print(files)

class DataLoader:
        
    def load_parquet(self, data_path):
        '''
        Loads the parquet file
        Returns the parquet file object
        '''
        try:
            self.parquet = pq.ParquetFile(data_path)
            return self.parquet
            logging.info("Parquet file loaded successfully")
        except Exception as e:
            logging.error(f"Error in loading parquet file: {str(e)}")
            raise CustomException(f"Error in loading parquet file: {str(e)}")
        
    def data_generator(self, chunck_size=10000):
        '''
        Returns the data Generator for the parquet file

        @param chunck_size: int, default=10000
        '''
        for i in self.parquet.iter_batches(batch_size=chunck_size):
            yield i.to_pandas()
    
    def num_rows(self):
        '''
        Return the number of rows in the parquet file
        '''
        return self.parquet.num_row_groups
    
class DataPreprocess:

    def preprocess(self, df):
        '''
        Preprocess the data
        1. Standardize the float data ['m0','pt']
        2. Convert the image data to 3D array
        3. Convert the data to 3D array
        4. Return the preprocessed data
        '''
        try:
            df1=[]
            # coverting the data into 3D array
            for a in df['X_jets']:
                df1.append(np.array([np.stack(a[0]),np.stack(a[1]),np.stack(a[2])]))
            df1 = np.array(df1)
            shape = df1.shape
            df1 = df1.reshape(shape[0],shape[2],shape[3],shape[1])
            # print(self.df['X_jets'].shape)
            df['X_jets'] = list(df1)
            del df1
            gc.collect()

            # Standardizing the float data ['pt', 'm0']
            mean_val = df[['m0','pt']].mean()
            std_val = df[['m0','pt']].std()
            df[['m0','pt']] = (df[['m0','pt']] - mean_val) / std_val
            logging.info("Data preprocessed successfully")
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            raise CustomException(f"Error in preprocessing data: {str(e)}")
        
        return df
    
    def split_data(self, df):
        '''
        Split the data into X and y
        1. X: [img_data, float_data]
        2. y: target variable
        '''
        img_data = np.array(df['X_jets'].tolist())
        float_data = df[['m0','pt']].values
        y = np.array(df.iloc[:,3])
        # print(img_data, float_data, y)
        return [img_data, float_data], y

class CustomModel:
    def __init__(self, input_shape_image=(125, 125, 3), input_shape_float=(2,)):
        '''
        Initialize the model
        '''
        try:
            # Define input for the image
            image_input = Input(shape=input_shape_image)

            # Define the CNN part
            x = Conv2D(64,(3,3), activation='relu', strides=(2,2))(image_input)
            x = Conv2D(64,(3,3), activation='relu', strides=(2,2))(x)
            x = Conv2D(32,(3,3), activation='relu', strides=(2,2))(x)
            x = Conv2D(32,(3,3), activation='relu')(x)
            x = Flatten()(x)

            # Define input for the two float features
            float_input = Input(shape=input_shape_float)

            # Concatenate the CNN output with the float inputs
            concatenated = concatenate([x, float_input])

            # Add a few dense layers for regression
            x = Dense(128, activation='relu')(concatenated)
            output = Dense(1, activation='sigmoid')(x)

            # Define the model with two inputs and one output
            self.model = Model(inputs=[image_input, float_input], outputs=output)

            self.model.compile(
                loss=tf.keras.losses.binary_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
            self.model.summary()
            # tf.keras.utils.plot_model(self.model, "model.png", show_shapes=True)
            logging.info("Model initialized successfully")
        except Exception as e:
            logging.error(f"Error in initializing model: {str(e)}")
            raise CustomException(f"Error in initializing model: {str(e)}")
    
    def train_model(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        '''
        Train the model
        '''
        self.history = self.model.fit(X, y,
                                 epochs=epochs, validation_split=validation_split, batch_size=batch_size
                                )
        # self.plot_history(epochs)
        logging.info("Model trained successfully")
        return self.history
    
    def plot_history(self, epochs):
        '''
        Plot the history of the model
        '''
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8,8))
        plt.subplot(1,2,1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1,2,2)
        plt.plot(range(epochs), loss, label="Training Loss")
        plt.plot(range(epochs), val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    def evaluate_model(self, X, y):
        '''
        Evaluate the model
        @param X: list, input data
        @param y: list, target variable
        '''
        return self.model.evaluate(X,y)
    
    def save_model(self):
        '''
        Save the model in different formats
        1. h5
        2. keras
        3. Tensorflow SavedModel
        '''
        next_model_version = max([int(i) for i in os.listdir("models") if i.isdigit()] + [0]) + 1
        path = f"models/{next_model_version}/model"
        self.model.save(path+".h5")
        self.model.save(path+".keras")
        self.model.export(path)
        logging.info(f"Model saved successfully on the path: {str(path)}")
        
        return path
    
    def load_model(self, path):
        '''
        Load the model from the given path
        Supports h5 and keras
        '''
        return load_model(path)

    def predict(self, X, y):
        '''
        Predict the target variable
        '''
        y_pred = self.model.predict(X)
        auc_score = roc_auc_score(y, y_pred)
        print("ROC Score: ", auc_score)
        # self.roc_curve(y, y_pred)

        return y_pred, auc_score
    
    def roc_curve(self, y, y_pred):
        '''
        Plot the ROC curve
        '''
        fpr, tpr, _ = roc_curve(y,  y_pred)
        auc_score = roc_auc_score(y, y_pred)

        #create ROC curve
        plt.plot(fpr,tpr,label="AUC="+str(auc_score))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()

class Utils:
    def load_model(self, path):
        '''
        Load the model from the given path
        Supports h5 and keras
        '''
        return load_model(path)
    
if __name__ == '__main__':
    obj = DataLoader()
    parquet = obj.load_parquet('data/'+files[0])
    data_gen = obj.data_generator(chunck_size=500)
    df = next(data_gen)
    # print(df.head())
    obj1 = DataPreprocess()
    df1 = obj1.preprocess(df)
    X, y = obj1.split_data(df1)
    print(X[1].shape, X[0].shape, y.shape)
    model = CustomModel()
    history = model.train_model(X, y, epochs=4)
    model.save_model()
    model.predict(X, y)
    gc.collect()
    
        
