import numpy as np

class KalmanFilter:
    def __init__(self, n, m):
        self.n = n  # state dimension
        self.m = m  # measurement dimension
        self.A = np.eye(n)  # state transition matrix
        self.H = np.zeros((m, n))  # measurement matrix
        self.Q = np.eye(n)  # process noise covariance
        self.R = np.eye(m)  # measurement noise covariance
        self.P = np.eye(n)  # error covariance
        self.x = np.zeros((n, 1))  # state estimate

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.n) - np.dot(K, self.H), self.P)

    def run(self, zs):
        xs = []
        for z in zs:
            self.predict()
            self.update(z)
            xs.append(self.x)
        return xs
