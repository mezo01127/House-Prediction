# House Price Prediction & Classification 🏠

This project implements a dual-approach machine learning system to analyze property data, estimate specific house prices, and classify them into market price categories using Python.

## 🚀 Key Features
* **Price Estimation:** Built a **Linear Regression** model to accurately predict house prices based on physical features like area, bedrooms, and bathrooms.
* **Price Classification:** Developed a **Logistic Regression** model to classify properties into market categories (Low, Medium, High).
* **Data Visualization:** Performed Exploratory Data Analysis (EDA) using **Seaborn** and **Matplotlib**, creating correlation heatmaps and pair plots to extract business insights.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (LinearRegression, LogisticRegression, train_test_split, metrics)

## 📊 Dataset Structure
The models are trained on a property dataset containing:
* `Area`: The square footage of the house.
* `Bedrooms` & `Bathrooms`: Number of rooms.
* `Price` / `Price_Category`: Targets for regression and classification tasks.

## 📈 Model Evaluation
* **Regression:** Evaluated using Mean Squared Error (MSE) and R2 Score to ensure predictive accuracy.
* **Classification:** Evaluated using Accuracy Score to track classification performance.
