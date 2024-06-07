from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


class SMOTE:

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))

        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1), return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        S = np.concatenate((self.X, S), axis=0)

        if self.is_dataframe:
            S = pd.DataFrame(S, columns=self.columns)

        return S

    def fit(self, X):
        self.is_dataframe = isinstance(X, pd.DataFrame)

        if self.is_dataframe:
            self.columns = X.columns
            self.X = X.values
        else:
            self.X = X

        self.n_minority_samples, self.n_features = self.X.shape

        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self