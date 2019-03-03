import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as st
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

iris_data = pd.read_csv("iris.in")

summary = iris_data.describe()
summary = summary.transpose()
print(summary.head())


iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species", palette = "husl")

plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
values = iris_data.values
X = values[:,0:4]
Y = values[:,4]

knn.fit(X,Y)
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
print('The accuracy of the SVM classifier on training data is {:.2f}'.format(knn.score(X, Y)))
