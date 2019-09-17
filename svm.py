# Support Vector Machine

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\VIPUL\\Downloads\\Polynomial-Linear-Regression-master (1)\\Polynomial-Linear-Regression-master\\Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
print(dataset.head())


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.svm import SVC
cls=SVC(kernel='linear',random_state=0)
cls.fit(X_train,y_train)
y_pred=cls.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
#Accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100,'%')
# Visualising the Logistic Regression results
from matplotlib.colors import ListedColormap
x_set,y_set= X_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1, stop=x_set[:,0].max()+1, step=0.01),
                   np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
plt.contour(x1,x2,cls.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_pred)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('TEST SET FOR LOGISTIC REGRESSION')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
