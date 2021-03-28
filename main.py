import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression

df = pd.read_csv('data.csv')
data = df.to_numpy()
X = data[:, 0:-1]
Y = data[:, -1]

model = LogisticRegression(learning_rate=0.0015, n_iterations=1000)
model.fit(X, Y)
print(model.weights)

plt.scatter((np.dot(X, model.weights.T)), Y)
plt.show()
