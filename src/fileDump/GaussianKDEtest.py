import numpy as np
from scipy import stats

data = [1, 2, 1, 4, 3, 2, 5, 2]

# 1. This returns a gaussian_kde object, NOT a list
kde = stats.gaussian_kde(data)

# 2. Evaluating it returns a NumPy ndarray
points = np.linspace(1, 5, 10)
density_array = kde(points)  # or kde.evaluate(points)

# 3. Convert the NumPy array to a standard Python list
density_list = density_array.tolist()

print(density_list)  # Output: <class 'list'>