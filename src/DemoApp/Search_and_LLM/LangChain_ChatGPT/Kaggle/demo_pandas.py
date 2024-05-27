
import pandas as pd

train = pd.read_csv("train.csv")

print(train)

# male, femaleを0, 1に変換
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

print(train)

# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Training
# x = train.loc[:, ["Pclass", "Sex", "SibSp", "Parch", "Fare"]]
# y = train["Survived"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

# model = LinearSVC(max_iter=10000000)
# model.fit(x_train, y_train)
# pred = model.predict(x_test)

# print(accuracy_score(y_test, pred))

print("################ Validation #################")


# Validation

test = pd.read_csv("test.csv")

print(test)

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

print(test)

# 欠損値の個数を確認
print(test.isnull().sum()) # fareに1つ欠損値があるので今回は使わない

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 「Fare」を除いて再度実行
x = train.loc[:, ["Pclass", "Sex", "SibSp", "Parch"]]
y = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

model = LinearSVC(max_iter=10000000)
model.fit(x_train, y_train)
pred = model.predict(x_test)

print(accuracy_score(y_test, pred))

x_testset = test.loc[:, ["Pclass", "Sex", "SibSp", "Parch"]]

pred = model.predict(x_testset)
print(pred)

print("len pred : ", len(pred))
print("#####")
print(y_test)
# print(x_testset)
