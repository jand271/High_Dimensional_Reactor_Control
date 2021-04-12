import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.plot(S)
plt.yscale("log")
plt.title("Singular Values")
plt.xlabel("Singular Value")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("singular_values.png")
