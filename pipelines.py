import math
from sklearn.metrics import roc_auc_score
from main_code import DataLoader, DataPreprocess, CustomModel, Utils
from logger import logging
from exception import CustomException
import os
import gc
# File locations
files = os.listdir('data')
print(files)

class Pipeline:
    def __init__(self):
        self.data = DataLoader()
        self.preprocess = DataPreprocess()
        self.utils = Utils()

    def run(self, data_path, chunck_size=1000, epochs=5, size=None):
        '''
        Run the pipeline to train the model
        @param data_path: str, path to the data
        @param chunck_size: int, default=1000
        @param epochs: int, default=5
        @param size: int, default=None
        '''
        try:
            self.model = CustomModel()
            if os.path.isfile(data_path):
                files = [os.path.basename(data_path)]
                data_path = os.path.dirname(data_path)
            elif os.path.isdir(data_path):
                files = os.listdir(data_path)

            accuracy_w = 0
            for i, file in enumerate(files):
                print(i+1, "/", len(files), " ", file)

                self.data.load_parquet(os.path.join(data_path, file))
                self.data_gen = self.data.data_generator(chunck_size)
                self.num_rows = self.data.num_rows()
                print("Number of records: ", self.num_rows)
                print("Chunck size: ", chunck_size)
                if size:
                    self.num_rows = min(self.num_rows, size)
                train_rows, test_rows = self.train_test_split(self.num_rows, test_size=0.1, chunck_size=chunck_size)
                print("Train batches: ", train_rows)
                print("Test batches: ", test_rows)

                print("Train: ")
                for i in range(train_rows):
                    print(i+1, "/", train_rows)
                    df = next(self.data_gen)
                    df = self.preprocess.preprocess(df)
                    X, y = self.preprocess.split_data(df)
                    self.model.train_model(X, y, epochs=epochs)
                    gc.collect()

                print("Test: ")
                avg_accuracy = 0
                for i in range(test_rows):
                    print(i+1, "/", test_rows)
                    df = next(self.data_gen)
                    df = self.preprocess.preprocess(df)
                    X, y = self.preprocess.split_data(df)
                    _, accuracy = self.model.predict(X, y)
                    avg_accuracy += accuracy
                if test_rows:
                    avg_accuracy /= test_rows
                    accuracy_w += avg_accuracy
                print("Average Accuracy: ", avg_accuracy)
                gc.collect()
            accuracy_w /= len(files)
            print("Accuracy: ", accuracy_w)

            self.model.save_model()
            del self.model
            gc.collect()
            logging.info("Trained successfully, Average Accuracy: %f", accuracy_w)
        except Exception as e:
            logging.info(f"Error in run pipeline: {str(e)}")
            raise CustomException(f"Error in run pipeline: {str(e)}")


    def train_test_split(self, num_rows, test_size=0.1, chunck_size=1000):
        '''
        Split the data into train and test
        @param num_rows: int, number of rows
        @param test_size: float, default=0.1
        @param chunck_size: int, default=1000
        '''
        iterations = math.ceil(num_rows / chunck_size)
        test_rows = math.ceil(iterations * test_size)
        train_rows = iterations - test_rows
        return train_rows, test_rows

    def predict(self, data_path, model_path, chunck_size=1000, size=None):
        '''
        Predict the model
        @param data_path: str, path to the data
        @param model_path: str, default=None
        @param chunck_size: int, default=1000
        @param size: int, default=100
        '''
        try:
            self.data.load_parquet(data_path)
            self.data_gen = self.data.data_generator(chunck_size)
            self.num_rows = self.data.num_rows()
            print("Number of records: ", self.num_rows)
            print("Chunck size: ", chunck_size)
            if size:    
                self.num_rows = min(self.num_rows, size)
            iterations = math.ceil(self.num_rows/chunck_size)
            model = self.utils.load_model(model_path)

            avg_roc_auc = 0
            for i in range(iterations):
                df = next(self.data_gen)
                df = self.preprocess.preprocess(df)
                X, y = self.preprocess.split_data(df)
                y_pred = model.predict(X)
                roc_score = roc_auc_score(y, y_pred)
                print("ROC Score: ", roc_score)
                avg_roc_auc += roc_score
            avg_roc_auc /= iterations
            print("Average ROC AUC: ", avg_roc_auc)
            gc.collect()
            logging.info("Predicted successfully, Average ROC AUC: %f", avg_roc_auc)
        except Exception as e:
            logging.info(f"Error in predict pipeline: {str(e)}")
            raise CustomException(f"Error in predict pipeline: {str(e)}")


if __name__ == '__main__':
    data_path = 'data'
    pipe = Pipeline()
    # pipe.run(data_path, epochs=5, chunck_size=10000)
    test_path = 'data/'+files[2]
    version = 8
    model_path = f'models/{version}/model.keras'
    pipe.predict(test_path, model_path,chunck_size=10000)

