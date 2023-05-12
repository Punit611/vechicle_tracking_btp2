from kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

# set up the Kalman filter
kf = KalmanFilter(n=2, m=1)
kf.A = np.array([[1, 1], [0, 1]])
kf.H = np.array([[1, 0]])
kf.Q *= 0.1
kf.R *= 10
kf.P *= 10

# generate some noisy measurements
np.random.seed(0)
zs = [np.array([i]) + np.random.randn() for i in range(50)]
print(zs)
# run the Kalman filter on the measurements
xs = kf.run(zs)

# plot the results
plt.plot([x[0, 0] for x in xs], label='Kalman filter')
plt.plot(zs, '.', label='measurements')
plt.legend()
plt.show()
