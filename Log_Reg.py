import numpy as np
import pandas as pd
import math

class logistic_regression:
    def __init__(self, training, validation, k, lr, tol):
        val = pd.read_csv(training, header=None, delimiter="\t")
        val2 = pd.read_csv(validation, header=None, delimiter="\t")

        self.a = lr
        self.t = tol
        self.k = k

        self.X = np.array(val.iloc[:, :-1])
        bias = np.ones(self.X.shape[0]).T
        bias = bias.reshape(len(bias), 1)
        self.X = np.hstack((bias, self.X))
        self.y = np.array(val.iloc[:, -1])

        self.w = [0] * len(self.X[0])
        self.w[0] = 0

        self.X2 = np.array(val2.iloc[:, :-1])
        bias = np.ones(self.X2.shape[0]).T
        bias = bias.reshape(len(bias), 1)
        self.X2 = np.hstack((bias, self.X2))
        self.y2 = np.array(val.iloc[:, -1])

        for i in self.w:
            if not(i == 1):
                i = 0

        self.w = np.array(self.w)

    def fit(self):

        prev_grad = 0

        for k in range (0, self.k-1):
            error = np.zeros(self.X.shape[1])

            for i in range(0, len(self.X)):
                error += (self.y[i] - self.sigmoid(np.dot(self.X[i], self.w)))*self.X[i]

            new_w= self.w+(self.a*error)

            if (np.linalg.norm(new_w-self.w) <= self.t):
                break
            self.w = new_w

            if ((k%100)==0):
                print(k)

        nf = open("weights.tsv", "w+")
        for i in self.w:
            nf.write(str(i) + "\n")

    def predict(self):
        weights = "weights.tsv"
        self.w = self.load_data(weights, None)


        pred = self.w.T.dot(self.X2.T)
        print("yay")
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred.at[i,j] = self.sigmoid(pred.at[i,j])


        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):

                if pred.iloc[i][j] < 0.5:
                    pred.iloc[i][j] = 0
                else:
                    pred.iloc[i][j] = 1

    def sigmoid(self, error):
        if error >= 0:
            f = math.exp(-error)
            x = 1 / (1 + f)
            return x
        else:
            f = math.exp(error)
            x = f / (1 + f)
            return x
        return x

    def load_data(self, path, header):
        df = pd.read_csv(path, header=header, delimiter="\t")
        return df

x = logistic_regression("train_dataset.tsv", "validation_dataset.tsv",
                        5000, 0.01, 0.0005)
x.fit()
x.predict()
