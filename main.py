import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd
from LogisticRegression import LogisticRegression

df = pd.read_csv('data.csv')
data = df.to_numpy()

# Normalise First 3 Columns
X = data[:, 0:-2]
X = normalize(X, axis=0)
X = np.hstack((X, data[:, 3:-1]))

n, m = X.shape
X = np.hstack((np.ones((n, 1)), X))

# Print Values For X
print(X)

Y = data[:, -1]

iterations = 100000

model = LogisticRegression(learning_rate=0.1, n_iterations=iterations)
cost_list = model.train(X, Y)

print(cost_list)

# print(model.costs)
print(f' Final Weights : {model.weights}')

plt.plot(np.arange(iterations), cost_list)
plt.show()
