"""
Created on Fri Dec 19 23:32:35 2025

@author: ashme
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

#Load Data
df = pd.read_csv(r"C:\Users\ashme\Downloads\Dataset for practice\Ca\laptopData.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

#Data Preprocessing
df.info()
df.head()
df.describe()
df.isnull().sum()
df.duplicated().sum()


# RAM
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').str.strip()
df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')
df['Ram'] = df['Ram'].fillna(df['Ram'].median()).astype(int)

# WEIGHT
df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').str.strip()
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Weight'] = df['Weight'].fillna(df['Weight'].median())

# PRICE
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Price'] = df['Price'].fillna(df['Price'].median())

# ENCODING
le = LabelEncoder()
df['Company'] = le.fit_transform(df['Company'])
df['TypeName'] = le.fit_transform(df['TypeName'])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#EDA

plt.figure(figsize=(6,4))
plt.hist(df['Price'], bins=20)
plt.title("Laptop Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['Ram'], df['Price'])
plt.xlabel("RAM (GB)")
plt.ylabel("Price")
plt.title("RAM vs Laptop Price")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['Weight'], df['Price'])
plt.xlabel("Weight (kg)")
plt.ylabel("Price")
plt.title("Weight vs Laptop Price")
plt.show()

corr = df[['Company','TypeName','Ram','Weight','Price']].corr()
plt.figure(figsize=(6,4))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

#Features
features = ['Company', 'TypeName', 'Ram', 'Weight']
X = df[features]
y = df['Price']

# Model 1: Linear Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nModel 1: LINEAR REGRESSION")
print("R2 Score:", r2_score(y_test, y_pred_lr))

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_lr)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# MODEL 2: K-MEANS CLUSTERING 
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(X)

wcss = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_cluster)
    wcss.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,7), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(6,4))
plt.scatter(df['Ram'], df['Price'], c=df['Segment'])
plt.xlabel("RAM")
plt.ylabel("Price")
plt.title("Laptop Segmentation using K-Means")
plt.show()

#PRICE CATEGORY 
df['Price_Category'] = pd.qcut(df['Price'], 3, labels=['Low','Medium','High'])
print("\nPrice Category Distribution:")
print(df['Price_Category'].value_counts())

# CLASSIFICATION DATA 
Xc = df[features]
yc = df['Price_Category']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

# MODEL 3: DECISION TREE
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    random_state=42
)
dt.fit(X_train_c, y_train_c)
y_pred_dt = dt.predict(X_test_c)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(y_test_c, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_dt))
print(classification_report(y_test_c, y_pred_dt))

ConfusionMatrixDisplay.from_predictions(
    y_test_c, y_pred_dt,
    display_labels=['Low','Medium','High'],
    cmap='Blues'
)
plt.title("Decision Tree – Confusion Matrix")
plt.show()

plt.figure(figsize=(18,8))
plot_tree(
    dt,
    feature_names=features,
    class_names=['Low','Medium','High'],
    filled=True,
    rounded=True
)
plt.show()

# MODEL 4: KNN
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train_scaled, y_train_c)
y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN")
print("Accuracy:", accuracy_score(y_test_c, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_knn))

ConfusionMatrixDisplay.from_predictions(
    y_test_c, y_pred_knn,
    display_labels=['Low','Medium','High'],
    cmap='Greens'
)
plt.title("KNN – Confusion Matrix")
plt.show()

# MODEL 5: NAIVE BAYES
nb = GaussianNB()
nb.fit(X_train_scaled, y_train_c)
y_pred_nb = nb.predict(X_test_scaled)

print("\nNAIVE BAYES")
print("Accuracy:", accuracy_score(y_test_c, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_nb))

ConfusionMatrixDisplay.from_predictions(
    y_test_c, y_pred_nb,
    display_labels=['Low','Medium','High'],
    cmap='Oranges'
)
plt.title("Naive Bayes – Confusion Matrix")
plt.show()

# FINAL COMPARISON 
summary = pd.DataFrame({
    'Model': [
        'Linear Regression (R2)',
        'Decision Tree',
        'KNN',
        'Naive Bayes'
    ],
    'Score': [
        r2_score(y_test, y_pred_lr),
        accuracy_score(y_test_c, y_pred_dt),
        accuracy_score(y_test_c, y_pred_knn),
        accuracy_score(y_test_c, y_pred_nb)
    ]
})

print("\nFINAL MODEL PERFORMANCE")
print(summary)

summary.set_index('Model').plot(kind='bar', figsize=(7,4))
plt.ylabel("Score")
plt.title("Model Comparison")
plt.show()

# SAVE OUTPUT
final_predictions = pd.DataFrame({
    'Actual_Category': y_test_c.values,
    'Predicted_Category': y_pred_dt
})

final_predictions.to_csv("laptop_price_predictions.csv", index=False)
print("\nPredictions saved successfully")