import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression


df = pd.read_csv('data.csv')
data = df.to_numpy()
X = data[:, 0:-1]
Y = data[:, -1]

regressor = LogisticRegression(lr=0.0015, n_iters=1000)
regressor.fit(X, Y)
print(regressor.weights)

plt.scatter((np.dot(X,regressor.weights.T)), Y)
plt.show()
