import pandas as pd
import matplotlib.pyplot as plt
data = {
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 4, 6, 8, 10]
}

df = pd.DataFrame(data)

print("Dataframe:")
print(df)

mean_x = df['X'].mean()
mean_y = df['Y'].mean()
print("\nMean of X:", mean_x)
print("Mean of Y:", mean_y)

plt.scatter(df['X'], df['Y'])
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()