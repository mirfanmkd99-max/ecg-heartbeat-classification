from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('data/mitbih_train.csv', header=None)
test = pd.read_csv('data/mitbih_test.csv', header=None)

X_train, y_train = train.iloc[:, :-
                              1].values, train.iloc[:, -1].astype(int).values
X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].astype(int).values

print(X_train.shape, X_test.shape)
print('train class counts: ', np.bincount(y_train))
print('test class counts: ', np.bincount(y_test))

# Check for NaN values
print(np.isnan(X_train).sum(), np.isnan(X_test).sum())

fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
for c in range(5):
    for i in range(3):
        axes[c].plot(X_train[y_train == c][i], alpha=0.7)
    axes[c].set_ylabel(f'class {c}')
plt.tight_layout()
plt.savefig('results/example_beats.png', dpi=150)


# split the data into training and validation sets
# stratify=y_train ensures that the class distribution is preserved in both training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
