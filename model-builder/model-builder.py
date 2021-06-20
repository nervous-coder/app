import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv', sep = ';')

df.drop(['release_date'], axis = 1, inplace = True)

a = []
for i in df['workload_ wc']:
    a.append(math.ceil(i))
df['workload_ wc'] = a

b = []
for i in df['pf_time']:
    b.append(math.ceil(i))
df['pf_time'] = b

data = df
X = data.drop(['workload_ wc'], axis = 1)
y = data['workload_ wc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)

model = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, min_samples_split = 10, min_samples_leaf = 1, max_features = 'auto', n_jobs = -1)
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))