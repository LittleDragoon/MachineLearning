import pandas as pd

# download data from https://archive.ics.uci.edu/ml/datasets/spambase
FILE_NAME = "spambase.data"


# Loads the CSV data
df = pd.read_csv(FILE_NAME, header=None)

# The first 57 columns are features
# The last column has the correct labels (targets)
X, y = df.iloc[:, :57].values, df.iloc[:, 57].values

print(X)
print(y)
