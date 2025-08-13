import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
data = sns.load_dataset('iris')
# print(data.isnull().sum()) # check null values in each column
# print(data.duplicated().sum()) #check duplicate rows 
data = data.drop_duplicates()
x = data.iloc[:,:4]
y = data.iloc[:,4:]
print(x.head())
# sns.boxplot(data=x)
# sns.pairplot(data,hue='species')
# petal_length and petal_width show clear separation — especially Setosa is very distinct.

# sepal_width overlaps a lot — not very helpful for classification.


# print(data.duplicated().sum())
# print(data.dtypes)
# print(data.describe(include='object')) # for categorical data it shows count freq unique top
# print(data.describe())
# plt.figure(figsize=(6,4))
# sns.boxplot(x=data['sepal_width'])  #showing dots outside boundry is outlier
# plt.grid(True)
# plt.show()

# Outlier Detection
def outlierss(data_sample):
 z = data_sample.describe()
 IQR = []
 lower = []
 upper = []
 for i in range(len(data_sample.columns)-1):
    iqr = z.iloc[6,i]-z.iloc[4,i]
    l = z.iloc[4,i]-1.5*iqr
    u = z.iloc[6,i]+1.5*iqr
    lower.append(l)
    upper.append(u)
    IQR.append(iqr)

 d = {
    'IQR' : IQR,
    'Lower': lower,
    'Upper' : upper
    }
 df = pd.DataFrame(d).T
 print(df)
 df.columns = ['sepal_length','sepal_width','petal_length','petal_width']


 for i in range(len(df.columns)):
    if z.iloc[3,i] < df.iloc[1,i] or z.iloc[7,i]>df.iloc[2,i]:
        print(df.columns[i])   # print outlier column


# Data Preprocessing

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # so it transform mean into 0 and std = 1 .it use z = x(i)-u/sigma
# When you apply StandardScaler, you're not changing the shape or distribution of the data, '
# 'you're just changing the scale of the axes (X and Y)
X_scaled = scaler.fit_transform(x)
X_scaled_df = pd.DataFrame(X_scaled,columns=x.columns)
X_scaled_df['species'] = y
X = X_scaled_df.drop('species',axis=1)
Y = X_scaled_df['species'].ffill()
# print(X_scaled_df)
# print(X_scaled_df.std(ddof=0))
# sns.pairplot(X_scaled_df,hue='species')
# plt.show()
# outlierss(X_scaled_df)

## Selecting A Model 

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold

# KNN

''' KNeighours classifier is lazy algorithum(supervised ml ) it does not learn from traing data and use traing data for calculating distance and gives smallest distance
if k(n) = 3, it means it gives 3 smallest distance and select class which has high frequency
it's mostly work on numeric data which should be scaled, so all feature have similar range in data we use standardScaler for transforming data
Example: 
suppose our traing data set contains 4 feature and 150 rows, so when new data or (row) is inserted 
it calculate distance = sqrt(sum((x1 - x2) ** 2))
Q1 = np.array([4.0, 3.8, 1.2, 0.2])
distance = np.sqrt(np.sum((P1 - Q1) ** 2))
and P1 is each row data of train data, so it calculate 150 distance and select 3 smallest distance and which class freq is more is selected for output
if 3 distance, in which 2 distance says class A and one class says B then final output will A
in case if decision is tie it calculate weights for each row with input data. weight = 1/distance 
so whose distance is small which has high weight and choose for final output

'''

# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train,y_train)
# y_pred = knn.predict(x_test)
# print(y_pred)
# print('Accuracy : ',accuracy_score(y_test,y_pred))
# print('\n Confusion Matrix : \n',confusion_matrix(y_test,y_pred))
# print('\n Classification report : \n',classification_report(y_test,y_pred))
# sns.heatmap(X_scaled_df.drop('species',axis=1).corr(),annot=True,cmap='coolwarm')
'''highly correlated or low correlated should be remove'''
# plt.show()


'''  Decision tree classifer 

1. Start at the Root Node
The algorithm begins with the entire dataset at the top (root).

It looks at all features (columns) and finds the best split.

2. Find the Best Split
For each feature and possible value, it tries:

“If I split the data here, how pure will the resulting groups be?”

Purity is measured using a metric like:

Gini Impurity (default in DecisionTreeClassifier)

or Entropy (information gain)

How Gini Impurity Works
Formula: 1-sigma(Pi)**2
  where Pi is the proportion of class i in a group.
  For Iris, classes = {Setosa, Versicolor, Virginica}



✅ Example:

Group with 3 Setosa and 0 others → Gini = 0 (pure)

Group with 1 Setosa, 1 Versicolor → Gini = 1 - (0.5² + 0.5²) = 0.5

🔁 The lower the Gini, the better the split.

3. Perform the Best Split
Once the best feature and value are found, split the data into 2 branches (left and right).

Each branch becomes a child node.

4. Repeat Recursively
Repeat steps 1-3 for each child node:

Try all features again (except previously used for strict trees).

Keep splitting until:

All data in a node is pure (same class), or

Max depth is reached, or

Minimum samples per node is reached

5. Leaf Node (Stopping Condition)
When a node can't be split further, it becomes a leaf node.

The label/class at a leaf node is the majority class of samples at that node.

6. Prediction
For a new data point:

Start at the root.

Follow the path: "Is petal length < 2.5? → Yes → Go left"

Continue until you hit a leaf → return that class.


'''

new_x = x.drop(['sepal_width','petal_width'],axis=1)
from sklearn.tree import DecisionTreeClassifier

x_train,x_test,y_train,y_test = train_test_split(new_x,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=3)
model.fit(x_train,y_train)
# sample_data = pd.DataFrame([[4.0,3.8,1.2,0.2]],columns=x_test.columns)
sample = [[4.0,1.2]]
# x_test = pd.concat([x_test,sample_data],ignore_index=True)
y_pred = model.predict(x_test)
# print(y_pred)
# model evolution
print('Accuracy : ',accuracy_score(y_test,y_pred))
print('\n Confusion Matrix : \n',confusion_matrix(y_test,y_pred))
print('\n Classification report : \n',classification_report(y_test,y_pred))

#cross validation
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
n=cross_val_score(model,x,y,cv=cv)
print(n,'\n',n.std())

'''Pruning means cutting back a decision tree to prevent it from becoming too complex (overfitting).
Why Prune?
A fully grown decision tree can:

Memorize training data (overfit)

Perform poorly on test/unseen data

Have many unnecessary branches that don’t help prediction

✅ Pruning simplifies the tree by removing those weak branches.

Term              	Meaning

Pruning	    =    Reducing the size of the tree to improve generalization
Pre-pruning  = 	Prevent growing large tree using max_depth, min_samples_split, etc
Post-pruning =	   Cut branches after full tree is built using ccp_alpha

'''