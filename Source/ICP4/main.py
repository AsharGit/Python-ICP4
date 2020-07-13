import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Read csv file and print correlation between sex and survived
train_df = pd.read_csv("train_preprocessed.csv")
print(train_df[["Sex", "Survived"]].groupby(["Sex"]).mean().sort_values(by="Sex"))


# Read csv file to assign to variables
df = pd.read_csv("glass.csv")
x_train = df.drop("Type", axis=1)
y_train = df["Type"]
# Training and testing of the data set
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# Naive Bayes classification
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
acc_svc = round(nb.score(x_train, y_train) * 100, 2)
# Print report
print("Naive Bayes accuracy is:", acc_svc)
print("Classification report:\n", classification_report(y_test, y_pred))

# Linear SVM classification
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
# Print report
print("svm accuracy is:", acc_svc)
print("Classification report:\n", classification_report(y_test, y_pred))

