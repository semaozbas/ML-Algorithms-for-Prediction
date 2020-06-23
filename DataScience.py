import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split # I
data = pd.read_csv("bank.csv")
df = pd.DataFrame(data)

df.info()
# print(data.describe())
#print(df.tail())
#print(df.columns)
# print(df.dtypes)

#############Vısualızatıon the data

plt.style.use('seaborn-darkgrid')
hist=df.hist(bins=10, figsize=(10,10), color='red',orientation='vertical') #A histogram is a representation of the distribution of data,Plot a histogram.
plt.show()

##########first way

# pd.DataFrame(data)["deposit"].replace(('yes', 'no'), (1, 0), inplace=True)
#
# deneme2=data.iloc[:,-1]#
# print(deneme2)


#########second way
# from sklearn.preprocessing import LabelEncoder
# job=data.iloc[:,1:2].values
# le=LabelEncoder()
# job[:,0]=le.fit_transform(job[:,0])
# print(job)


###########third way
categorical_feature_mask = data.dtypes==object
categorical_cols = data.columns[categorical_feature_mask].tolist()
le = LabelEncoder()#le is a object
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
#print(data)

X=data.iloc[:,0:16]
y=data.iloc[:,-1]

tree_classifier = DecisionTreeClassifier()#DecisionTreeClassifier
tree_scores = cross_val_score(tree_classifier, X, y, cv=5,verbose=0)

#################WITH ABOVE CODE WE DONT NEED A FOR LOOP FOR ITERATIONS OF KFOLD
# for train_index, test_index in kf.split(X):
#     # Split train-test
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y[train_index], y[test_index]

print(tree_scores)
_sum = sum(tree_scores)
print("Decision_Tree_accurancy:", _sum/5)

##############Classıfıcatıon report for decision tree
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# tree_classifier.fit(X_train,y_train)
# y_pred=tree_classifier.predict(X_test)
# results = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix :')
# print(results)
# # print ('classification_report for Decision Tree :')
# print(classification_report(y_test, y_pred))

rnd=RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rnd.fit(X_train,y_train)
y_pred=rnd.predict(X_test)
print("Random_Forest_accurancy:",metrics.accuracy_score(y_test, y_pred))
score = r2_score(y_test, y_pred)
print("Random_Forest_r2score:", score)


##############Classıfıcatıon randomForest
rnd=RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rnd.fit(X_train,y_train)
y_pred=rnd.predict(X_test)
results = confusion_matrix(y_test, y_pred)
print('Confusion Matrix for Random Forest :')
print(results)
print('classification_report for Random Forest :')
print(classification_report(y_test, y_pred))

