import pandas as pd

df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv')

X = df[['Experience Years']]
X.shape

y = df['Salary']
y.shape

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2529)
X_train

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)
