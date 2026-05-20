import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,mean_absolute_error

df=pd.read_csv("/content/House Price Prediction Dataset.csv")

df.head()

"""##B. Choose:
o Features (X): columns used for prediction
o Target (y): the thing to predict
"""

x=df[["Id","Area","Bedrooms",'Bathrooms',"Floors","YearBuilt","Location","Condition","Garage"]]

y=df["Price"]

"""Step C —Data visualization (must-have for your project)
Do at least 2–3 plots, for example:

1. Scatter plot (feature vs target)
"""

plt.scatter(x=df["Floors"],y=df["Price"])
plt.xlabel("Floors")
plt.ylabel("Price")
plt.title("Floors vs Price")
plt.show()

plt.scatter(x=df["Area"],y=df["Price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

"""2. Correlation heatmap"""

cor=df.drop(["Location","Condition","Garage"],axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap="coolwarm",linewidths=0.6)
plt.title("Matrix Heatmap")
plt.show()

"""3.bar column"""

plt.bar(df["Garage"],df["Price"])
plt.xlabel("Garage")
plt.ylabel("Price")
plt.title("	Garage vs Price")
plt.show()

plt.bar(df["Condition"],df["Price"])
plt.xlabel("Condition")
plt.ylabel("Price")
plt.title("	Condition vs Price")
plt.show()

"""LabelEncoder"""

le=LabelEncoder()

df['Garage']=le.fit_transform(df['Garage'])
df['Condition']=le.fit_transform(df['Condition'])
df['Location']=le.fit_transform(df['Location'])

df.info()

"""Step D —Train-test split
Split into train and test sets (e.g., 80/20).
"""

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)

"""Step E —Train Linear Regression
Fit the model on training data.
"""

model1=LinearRegression()

model1.fit(x_train,y_train)

"""Step F —Predict on test data
Generate predictions and compare.
"""

y_pre=model1.predict(x_test)
y_pre

"""Step F —Predict on test data
Generate predictions and compare.
"""

MAE=mean_absolute_error(y_test,y_pre)
MSE=mean_squared_error(y_test,y_pre)
R2=r2_score(y_test,y_pre)

print(f"mean_absolute_error is :", MAE)
print(f"mean_squared_error is :", MSE)
print(f"r2_score is :", R2)

"""Step H —Final visualization + conclusion
 Plotactual vs predicted
 Writea short conclusion: whether the model performs well and limitations
"""

plt.plot(y_test, label="actual")
plt.plot(y_pre, label="predicted")
plt.legend()
plt.show()

"""### Conclusion

The Linear Regression model trained on this dataset shows very poor performance, as indicated by a negative R-squared score (-0.034). An R-squared value close to 0 or negative means that the model does not explain the variability of the target variable (Price) any better than simply predicting the mean. The high Mean Absolute Error (MAE) of approximately 243,548 and Mean Squared Error (MSE) of approximately 7.98 x 10^10 further confirm that the model's predictions are far from the actual values.

#### Limitations and Next Steps:

1.  **Feature Engineering**: The current features might not be sufficiently informative. Creating new features from existing ones (e.g., age of the house from `YearBuilt`, combining `Bedrooms` and `Bathrooms`) could improve the model.
2.  **Categorical Feature Handling**: While Label Encoding was used, it might not be the best approach for 'Location', 'Condition', and 'Garage' if there's no inherent ordinality. One-hot encoding might be more appropriate.
3.  **Model Choice**: Linear Regression assumes a linear relationship between features and the target. Given the poor performance, a different model (e.g., RandomForestRegressor, GradientBoostingRegressor) or a more complex linear model might be necessary.
4.  **Data Scaling**: Features like 'Area' and 'YearBuilt' have very different scales. Scaling the features (e.g., using `StandardScaler` or `MinMaxScaler`) could help some models perform better.
5.  **Outliers**: The scatter plots showed some potential outliers. Investigating and handling outliers could improve model robustness.

Overall, further data preprocessing, feature engineering, and experimentation with different regression models are recommended to improve prediction accuracy.
"""
