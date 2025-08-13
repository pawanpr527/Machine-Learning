# Titanic Survival Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

read_data = pd.read_csv('files/train.csv')
test_data = pd.read_csv('files/test.csv')

print(read_data.shape)
print(read_data.head())
print(read_data.tail())
print(read_data.isnull().sum())

# it can be use as groupby
female_data_age = read_data[read_data['Sex']=='female']['Age']

# Fill missing Age values with mean Age based on Sex (group-wise)

read_data['Age'] = read_data['Age'].fillna(read_data.groupby('Sex')['Age'].transform('median'))
test_data['Age'] = test_data['Age'].fillna(test_data.groupby('Sex')['Age'].transform('median'))



# drop cabin bcoz of so many missing values
read_data = read_data.drop('Cabin',axis=1)
test_data = test_data.drop('Cabin',axis=1)

#fill Embarked column with most freuent value
read_data['Embarked'] = read_data['Embarked'].fillna(read_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])


# drop irrelevent column like name or ticket
read_data = read_data.drop(columns=['Name','Ticket'])
test_data = test_data.drop(columns=['Name','Ticket'])

#Encoding our categorical data using map

read_data['Sex'] = read_data['Sex'].map({
    'male' : 0,
    'female' : 1
})
read_data['Embarked'] = read_data['Embarked'].map({
    'C' : 0,
    'Q' : 1,
    'S' : 2
})
test_data['Sex'] = test_data['Sex'].map({
    'male' : 0,
    'female' : 1
})
test_data['Embarked'] = test_data['Embarked'].map({
    'C' : 0,
    'Q' : 1,
    'S' : 2
})

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_psg_id = test_data['PassengerId']
#Encoding our categorical data using LabelEncoder

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# read_data['Sex'] = le.fit_transform(read_data['Sex'])
# read_data['Embarked'] = le.fit_transform(read_data['Embarked'])

# print(read_data.iloc[:10,3:])



# y = read_data.iloc[:,1]
# print(y.shape)
# print(y.value_counts()) # count how many survived or not

# Visualization

# 1. Survival Count

# sns.countplot(x='Survived',data=read_data)
# plt.show()

# 2. Survival by Gender

# sns.countplot(x='Sex',hue='Survived',data=read_data)
# plt.show() #it shows more female survived than mens

# 3. Survival by Passenger Class

# sns.countplot(x='Embarked',hue='Survived',data=read_data)
# plt.show()

# 4. Age distribution with KDE

# sns.histplot(data=read_data,x='Age',kde=True,hue='Survived',bins=30)
# plt.show() #Tells you age groups of survivors and non-survivors.

# 5. Correlation 

# corr = read_data.corr(numeric_only=True)
# sns.heatmap(corr,annot=True,cmap='coolwarm')
# plt.show()

# meddile line shows median of ages which survived or not

# sns.boxplot(y='Age',x='Survived',data=read_data)
# plt.show()


#model building

#  Logistic Regression

x = read_data.drop(columns=['Survived','PassengerId'],axis=1)
test_x = test_data.drop(columns=['PassengerId'],axis=1)

y = read_data['Survived']

# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# model = LogisticRegression(class_weight='balanced')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# print(accuracy_score(y_test,y_pred),'\n',confusion_matrix(y_test,y_pred),'\n',classification_report(y_test,y_pred))
# print(cross_val_score(model,x,y,cv=10))

''' Linear model =  Logistic Regression assumes a linear relationship between features and the target. 
But Titanic data has non-linear patterns (e.g., survival depends on combinations like Sex + Pclass).'''

# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42)
# from sklearn.model_selection import GridSearchCV

# params = {
#     'n_estimators': [100, 200],
#     'max_depth': [5, 10, 20],
#     'min_samples_split': [2, 4],
# }
model.fit(x_train,y_train)
# grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
# grid.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(cross_val_score(model,x,y,cv=5))

# submission = pd.DataFrame({
#     'PassengerId' : test_psg_id,
#     'Survived' : y_pred
# })

# submission.to_csv('submission.csv',index=False)

# index = []
# series = dict()
# for i in y_test.index:
#     index.append(i)
# y_pred_series = pd.Series(y_pred,index=index)

# for i in index:
#     if y_test[i]!=y_pred_series[i]:
#         series[i] = y_pred_series[i]
# # print(series)
# temp = read_data[read_data['PassengerId'].isin(list(series.keys()))]
