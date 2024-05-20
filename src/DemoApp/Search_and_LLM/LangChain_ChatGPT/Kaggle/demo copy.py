
import pandas as pd

train = pd.read_csv("DemoTrain.csv")

print(train)
print(train.loc[:, ["time", "action"]])
# print(train.loc[["action"]])

# # from sklearn.svm import LinearSVC
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score

# # # Training
# # x = train.loc[:, ["time", "action"]]
# # y = train["tool"]

# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

# # model = LinearSVC(max_iter=10000000)
# # model.fit(x_train, y_train)
# # pred = model.predict(x_test)

# # print(accuracy_score(y_test, pred))

# print("################ Validation #################")


# # Validation

# test = pd.read_csv("DemoTest.csv")

# print(test)

# # # 欠損値の個数を確認
# # print(test.isnull().sum()) # fareに1つ欠損値があるので今回は使わない

# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Training
# x = train.loc[:, ["time", "action"]]
# y = train["tool"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

# model = LinearSVC(max_iter=10000000)
# model.fit(x_train, y_train)
# pred = model.predict(x_test)

# print(accuracy_score(y_test, pred))

# # x_testset = test.loc[:, ["Pclass", "Sex", "SibSp", "Parch"]]
# x_testset = test.loc[:, ["time", "action"]]

# pred = model.predict(x_testset)
# print(pred)
