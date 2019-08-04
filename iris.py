from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing

##one way to load dataset
# X,y=load_iris(return_X_y=True)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)


####DATA VISUALIZATION
print(dataset.head())
print("#################################################################")
print("#################################################################")
print()
print("DATASET DESCRIPTION")
print(dataset.describe())
print("#################################################################")
print("#################################################################")
print()

dataset.hist()
plt.show()

###TRAIN TEST SPLIT###

array=dataset.values
X=array[:,0:4]
y=array[:,4]
X_train,X_val,Y_train,Y_val=model_selection.train_test_split(X,y,test_size=0.20)



######MODELS###########

###LOGISTIC REGRESSION#######


# solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’).
# Algorithm to use in the optimization problem.

# For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
# For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
# ‘liblinear’ and ‘saga’ also handle L1 penalty
# ‘saga’ also supports ‘elasticnet’ penalty
# ‘liblinear’ does not handle no penalty

# multi_class : str, {‘ovr’, ‘multinomial’, ‘auto’}, optional (default=’ovr’)
# If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, Y_train)
print("LOGISTIC REGRESION SCORE",clf.score(X_val,Y_val))

####RANDOM FOREST#####

Random_forest = RandomForestClassifier(
		n_jobs=-1,
		criterion='entropy',
		n_estimators=100,
		max_features=.33,
		max_depth=30,
		min_samples_leaf=3,
		# min_samples_split=3,
		# max_leaf_nodes=35000,
		warm_start=True,
		oob_score=True,
		random_state=321)

Random_forest.fit(X_train,Y_train)

print("RANDOM FOREST SCORE",Random_forest.score(X_val,Y_val))



### K nearest neigbour########

K_neighbor = KNeighborsClassifier(n_neighbors=3)
K_neighbor.fit(X_train,Y_train)
print("K nearest neighbour SCORE",K_neighbor.score(X_val,Y_val))


###Support Vector Machine###

Support_vector_machine = svm.SVC(gamma='auto', decision_function_shape='ovo')

Support_vector_machine.fit(X_train,Y_train)

print("Support vector machine SCORE",Support_vector_machine.score(X_val,Y_val))