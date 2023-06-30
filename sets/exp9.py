import numpy as np
import matplotlib.pyplot as plt
from moepy import lowess
x = np.linspace(0, 5, num=150)
y = np.sin(x)+(np.random.normal(size=len(x)))/10
lowess_model = lowess.Lowess()
lowess_model.fit(x, y)
x_pred = np.linspace(0, 5, 26)
y_pred = lowess_model.predict(x_pred)
plt.plot(x_pred, y_pred, '--', label='Lowess', color='r', zorder=3)
plt.scatter(x, y, label='Noisy sin wave', color='m', s=10, zorder=1)