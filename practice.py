import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = x

plt.figure()
plt.plot(x, y)

# plt.semilogy(x, y)

plt.show()