from sklearn.datasets import fetch_openml
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.ensemble import RandomForestClassifier


titanic = fetch_openml('titanic', version = 1, as_frame = True)

df = titanic.frame 

print(df.head())
print(df.columns.tolist())
print()
print(df.isnull().sum())

df = df.drop(columns=['cabin', 'boat', 'body', 'home.dest', 'name', 'ticket'])

df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])


print(df.isnull().sum())

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(df.head())

X = df.drop(columns=['survived'])
y = df['survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

