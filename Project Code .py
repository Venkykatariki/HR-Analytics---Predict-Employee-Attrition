#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# Exploring a dataset from a multinational consultancy firm to analyze and predict employee attrition based on various attributes collected over a certain period.

# # 2. Importing Necessary libraries

# In[15]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from scipy.stats import chi2_contingency
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
palette_colors = sns.color_palette("tab10")
color1 = palette_colors[0] 
color2 = palette_colors[1]
colors = [color1, color2]


# # 3. Load Dataset
# Importing our Dataset and setting the Employee Number as the index

# In[16]:


df = pd.read_csv("C:/Users/venka/Downloads/HR Data.xlsx - HR data.csv")
df


# Drop rows with critical missing values

# In[22]:


df = df.dropna(subset=['Department', 'Monthly Income', 'Years At Company', 'Years Since Last Promotion'])


# Convert categorical columns to category type

# In[24]:


categorical_cols = ['Department', 'Attrition', 'Gender', 'Job Role']
for col in categorical_cols:
    df[col] = df[col].astype('category')


# Ensure numerical columns are correct type

# In[30]:


numerical_cols = ['Monthly Income', 'Years At Company', 'Years Since Last Promotion', 'Age']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Handle outliers in Monthly Income (cap at 99th percentile)

# In[31]:


income_cap = df['Monthly Income'].quantile(0.99)
df['Monthly Income'] = df['Monthly Income'].clip(upper=income_cap)


# # 4. Preliminary Data Analysis
# Data Dimensions / Data Types / Missing Values

# In[25]:


df.shape


# In[26]:


df.duplicated().any()


# In[27]:


df.info()


# # 5. Exploratory Data Analysis (EDA)

# Set the max rows and columns to display

# In[28]:


pd.options.display.max_rows = len(df)
pd.options.display.max_columns = len(df.columns)
df.describe()


# --- Summary Statistics ---

# In[35]:


print("\nSummary Statistics by Department:")
summary_stats = df.groupby('Department').agg({
    'Monthly Income': ['mean', 'median', 'std'],
    'Years At Company': ['mean', 'median'],
    'Years Since Last Promotion': ['mean', 'median'],
    'Attrition': lambda x: (x == 'Yes').mean() * 100
}).round(2)
print(summary_stats)


# ## 5.1   Department-wise Attrition Rate

# In[43]:


dept_attrition_counts = df[df['Attrition'] == 'Yes']['Department'].value_counts()
dept_total_counts = df['Department'].value_counts()
dept_attrition_rate = (dept_attrition_counts / dept_total_counts * 100).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=dept_attrition_rate.index, y=dept_attrition_rate.values, palette='coolwarm')
plt.title('Department-wise Attrition Rate (%)')
plt.ylabel('Attrition Rate (%)')
plt.xlabel('Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # 5.2 Salary Band vs Attrition

# In[44]:


# Create salary bands using quartiles
df['Salary Band'] = pd.qcut(df['Monthly Income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Calculate attrition rate by salary band
salary_attrition = pd.crosstab(df['Salary Band'], df['Attrition'], normalize='index') * 100

# Plotting attrition rate for each salary band
salary_attrition['Yes'].plot(kind='bar', color='orange', figsize=(8, 5))
plt.title('Attrition Rate by Salary Band')
plt.ylabel('Attrition Rate (%)')
plt.xlabel('Salary Band')
plt.xticks(rotation=0)
plt.ylim(0, 50)
plt.tight_layout()
plt.show()


# # 5.3 Promotions (Years Since Last Promotion vs Attrition)

# In[45]:


plt.figure(figsize=(8, 5))
sns.boxplot(x='Attrition', y='Years Since Last Promotion', data=df, palette='pastel')
plt.title("Years Since Last Promotion vs Attrition")
plt.xlabel("Attrition")
plt.ylabel("Years Since Last Promotion")
plt.tight_layout()
plt.show()


# # Result

# In[5]:


# Load the data
df = pd.read_csv("C:/Users/venka/Downloads/HR Data.xlsx - HR data.csv")  # Adjust the filename if needed

# Drop irrelevant columns
df.drop(['Employee Number', 'emp no'], axis=1, errors='ignore', inplace=True)

# Encode target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical features
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# # 6  classification model (Logistic Regression)

# Import required libraries

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


# Load data

# In[10]:


df = pd.read_csv("C:/Users/venka/Downloads/HR Data.xlsx - HR data.csv")


# # 6.1 Confusion Matrix and ROC Curve

# In[12]:


cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)


# In[13]:


# Plot Confusion Matrix & ROC Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title("Confusion Matrix")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Expected")

# ROC Curve
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()


# # 7 Decision Tree Classification Model

# In[16]:


# Build and train the Decision Tree model
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree - Confusion Matrix")
plt.show()


# In[18]:


# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[20]:


# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Expected")

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# In[21]:


# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree")
plt.show()


# # 8 SHAP value analysis

# In[48]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load data (use read_csv for .csv files)
try:
    data = pd.read_csv(r"C:\Users\venka\Downloads\HR Data.xlsx - HR data.csv")
except FileNotFoundError:
    print("Error: File not found. Please check the file path and name.")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Drop irrelevant columns
data = data.drop(columns=['emp no', 'Employee Number', 'Over18', 'Employee Count', 'Standard Hours', 'CF_attrition label'], errors='ignore')

# Encode target
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

# Features and target
X = data.drop(columns=['Attrition'])
y = data['Attrition']

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


# SHAP values for first test instance
shap_df = pd.DataFrame({'Feature': X_test.columns, 'SHAP Value': shap_values[0]})
print(shap_df.sort_values(by='SHAP Value', ascending=False))


# In[49]:


plt.figure()
shap.summary_plot(shap_values, X_test)
plt.savefig('shap_summary.png')
plt.close()

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
plt.savefig('shap_force.png')
plt.close()

