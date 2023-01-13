import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from transformer import Encode_Transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('data.csv')

df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 100]
df = df[df["person_income"] <= 4000000]

features = ['person_income',
            'person_home_ownership',
            'person_emp_length',
            'loan_intent',
            'loan_grade',
            'loan_percent_income',
            'cb_person_default_on_file']
target = 'loan_status'

X = df[features]
y = df[target]
transformer = Encode_Transformer()
X = transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20, stratify=y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Test", '\n',classification_report(y_test, model.predict(X_test)))
print('Train', '\n',classification_report(y_train, model.predict(X_train)))

pickle.dump(model, open("model.pkl", "wb"))