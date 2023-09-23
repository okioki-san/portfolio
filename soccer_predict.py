from pandas import Series,DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_excel("FC東京_data.xlsx", sheet_name="target")

df_X = df.copy()
df_Y = df.copy()


df_X = df_X.drop(["節", "天候", "結果"], axis=1)
df_Y = df_Y["結果"]

x_train, x_test, y_train, y_test = split(df_X, df_Y, train_size=0.8, test_size=0.2, shuffle=False)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_pred)
print(metrics.accuracy_score(y_test, y_pred))