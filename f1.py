1. Write a Python program to compute the following descriptive statistics for both Study_Hours and Exam_Score:

 Mean

 Median

 Standard Deviation (6 Marks)
Assume that the Exam_Score follows a normal distribution.

b)Write a Python program to calculate the probability that a randomly selected student scores more than 75 marks. (2 Marks)

c) Plot a histogram for Exam_Score and interpret whether the distribution appears normal or skewed. (2 Marks)

d) Based on the statistical results, briefly explain the relationship between study hours and exam performance. (2 Marks)


import numpy as np
from scipy.stats import norm

study_hrs = np.array([2, 4, 6, 8, 10, 12, 14])
score = np.array([45, 50, 60, 70, 78, 85, 90])

print("Study Hours")
print("Mean:", np.mean(study_hrs))
print("Median:", np.median(study_hrs))
print("Std Dev:", np.std(study_hrs))

print("\nExam Scores")
print("Mean:", np.mean(score))
print("Median:", np.median(score))
print("Std Dev:", np.std(score))


#b)Assume that the Exam_Score follows a normal distribution. Write a Python program to calculate the probability
#that a randomly selected student scores more than 75 marks. (2 Marks)
prob = 1 - norm.cdf(75, np.mean(score), np.std(score))
print("\nProbability of scoring more than 75:", prob)
#This Python code snippet calculates the probability that a randomly selected score is greater than 75, assuming the scores follow a normal distribution.
#Here is a breakdown of the formula:
#np.mean(score): Calculates the average value of the data set.
#np.std(score): Calculates the standard deviation (spread) of the data set.
#norm.cdf(75, mean, std): scipy.stats.norm.cdf calculates the Cumulative Distribution Function.
#This function returns the probability that a value is less than or equal to 75.
#1 - ...: Since the total probability under a normal curve is 1, subtracting the CDF result from 1 gives the upper tail: the probability that a value is greater than 75.
#c) Plot a histogram for Exam_Score and interpret whether the distribution appears normal or skewed. (2 Marks)
import matplotlib.pyplot as plt
import seaborn as sns
#sns.histplot(score, bins=7, kde=True, color='lightgreen', edgecolor='red')
plt.hist(score, bins=5,edgecolor='black')
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.title("Histogram of Exam Scores")
plt.show()
While the small sample size makes a perfect "bell curve" difficult to see, the distribution is not significantly skewed.

Normal: The data points are spread relatively evenly around the center (70), with distances to the edges (45 and 90) being similar.


Skewed: If the data had a "tail" (e.g., if there were a few very low scores like 10, or very high scores like 150), it would be considered skewed
#d) RELATIONSHIP BETWEEN STUDY HOURS AND MARKS
plt.figure(figsize=(8, 6))
plt.scatter(study_hrs, score, color='red', alpha=0.6, s=100) # s is marker size

# Add title and labels
plt.title('Relationship Between Study Hours and Exam Marks')
plt.xlabel('Study Hours')
plt.ylabel('Exam Marks')
plt.grid(True) # Add a grid for better readability

# Display the plot
plt.show()



2. A college has collected data on student academic performance to analyze trends and patterns
using data visualization techniques.
Dataset (Given):
a) Write a Python program to create a bar chart showing the average internal marks department-wise. (3 Marks)

b) Create a scatter plot to visualize the relationship between attendance percentage and internal marks. (3 Marks)

c) Draw a box plot for Internal_Marks to identify spread and outliers, and briefly comment on the distribution. (3 Marks)

d) Write a Python program to create a heatmap showing student performance across subjects.. (3 Marks)

import pandas as pd
import matplotlib.pyplot as plt
data = {
    "Student_ID": ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"],
    "Department": ["CSE","ECE","ME","CSE","EEE","CSE","ME","ECE","CSE","CSE"],
    "Attendance": [85,92,78,70,88,75,68,80,45,32],
    "Internal_Marks": [95,85,65,82,70,90,60,75,10,15]
}

df = pd.DataFrame(data)
#a) Write a Python program to create a bar chart showing the average internal marks department-wise. (3 Marks)
dept_avg = df.groupby("Department")["Internal_Marks"].mean()

plt.bar(dept_avg.index, dept_avg.values)
plt.xlabel("Department")
plt.ylabel("Average Internal Marks")
plt.title("Average Internal Marks by Department")
plt.show()

#b) Create a scatter plot to visualize the relationship between attendance percentage and internal marks. (3 Marks)
plt.scatter(df["Attendance"], df["Internal_Marks"])
plt.xlabel("Attendance (%)")
plt.ylabel("Internal Marks")
plt.title("Attendance vs Internal Marks")
plt.show()

#c) Draw a box plot for Internal_Marks to identify spread and outliers, and briefly comment on the distribution. (3 Marks)
plt.boxplot(df["Internal_Marks"])
plt.ylabel("Internal Marks")
plt.title("Box Plot of Internal Marks")
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create sample student data
data = {
    'Student': ['Smitha', 'Raj', 'Ayushi', 'Aarav', 'Poorna'],
    'Math': [85, 40, 75, 90, 60],
    'Science': [90, 50, 80, 85, 70],
    'English': [70, 60, 90, 75, 80],
    'History': [80, 30, 60, 80, 90]
}

df = pd.DataFrame(data)
df.set_index('Student', inplace=True)

# 2. Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# 3. Create the heatmap using seaborn
# 'annot=True' displays the data value in each cell
# 'cmap' defines the color palette
sns.heatmap(df,  cmap='YlGnBu', annot=False,fmt='d', linewidths=.5)

# 4. Add titles and labels
plt.title('Student Performance Heatmap')
plt.xlabel('Subjects')
plt.ylabel('Students')

# 5. Display the plot
plt.show()


3. Using the given dataset, apply Linear Regression to predict a student’s Internal Marks based
on Study Hours and Attendance. Apply Logistic Regression to classify whether a student will
be placed or not
Write a Python program to load the dataset and visualize the relationship between:
 Study Hours vs Internal Marks
 Attendance vs Internal Marksa

a) Build a Multiple Linear Regression model using Python to predict Internal_Marks. Predict the internal marks of a student who studies 7 hours with 80% attendance.
Evaluate the model using Mean Squared Error (MSE) and briefly interpret the result. (6 marks)

Perform data preprocessing and visualize the relationship between Internal Marks and Placement status using a suitable plot.


b) Build a Logistic Regression model to predict the Placed variable. (6 marks)
Predict whether a student with the following details will be placed:
 Study Hours = 6
 Attendance = 78%
 Internal Marks = 68
 Evaluate the model using accuracy score and explain its significance.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

data = {
    "study": [2,4,6,8,10,5,7,9],
    "attend": [65,70,75,80,90,72,78,85],
    "marks": [45,55,65,78,90,60,70,85],
    "placed": [0,0,1,1,1,0,1,1]
}

df = pd.DataFrame(data)
#Study Hours vs Internal Marks

plt.scatter(df["study"], df["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Internal Marks")
plt.show()


#Attendance vs Internal Marks
plt.scatter(df["attend"], df["marks"])
plt.xlabel("Attendance")
plt.ylabel("Internal Marks")
plt.show()


#Build a Multiple Linear Regression model using Python to predict Internal_Marks.
#Predict the internal marks of a student who studies 7 hours with 80% attendance.
#Evaluate the model using Mean Squared Error (MSE) and briefly interpret the result.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1.1 Create Sample Dataset
data = {
    'Study_Hours':[2, 4, 6, 8, 10, 5, 7, 9],
    'Attendance': [65, 70, 75, 80, 90, 72, 78, 85],
    'Internal_Marks': [45, 55, 65, 78, 90, 60, 70, 85],
    'Placed': [0,0,1,1,1,0,1,1]
}
df = pd.DataFrame(data)

# 1.2 Define Features (X) and Target (y)
X = df[['Study_Hours', 'Attendance']]
y = df['Internal_Marks']

# 1.3 Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1.4 Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 1.5 Predict Internal Marks for 7 hours study and 80% attendance
new_data = np.array([[7, 80]])
prediction = model.predict(new_data)
print(f"Predicted Internal Marks (7h study, 80% attendance): {prediction[0]:.2f}")

# 1.6 Evaluation (Mean Squared Error)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")


#Perform data preprocessing and visualize the relationship between Internal Marks and
#Placement status using a suitable plot.
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.boxplot(x='placed', y='marks', data=df)
plt.title('Internal Marks vs Placement Status')
plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.show()

#b) Build a Logistic Regression model to predict the Placed variable. (6 marks)
#Predict whether a student with the following details will be placed:
# Study Hours = 6
# Attendance = 78%
# Internal Marks = 68
# Evaluate the model using accuracy score and explain its significance.

# Features (Independent variables)
X = df[['Study_Hours', 'Attendance', 'Internal_Marks']]
# Target (Dependent variable)
y = df['Placed']

# 2. Build the Logistic Regression Model
# Using default solver 'lbfgs' suitable for small datasets
model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully.")
# 3. Predict for a new student
# Study Hours = 6, Attendance = 78%, Internal Marks = 68
new_student = [[6, 78, 68]]
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print(f"\nPrediction for new student (Study=6, Att=78%, Mark=68):")
print(f"Placed: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability of Placement: {probability[0][1]:.2f}")

# 4. Evaluate the model using accuracy score
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


4. An email service provider wants to classify emails as Spam or Not Spam using the Naïve Bayes algorithm.
a) Identify the features and target variable, and briefly explain why Naïve Bayes is suitable for this problem. (2 Marks)

b) Write a Python program to preprocess the dataset by encoding categoricalvalues into numerical form. (3 Marks)

c) Implement a Naïve Bayes classifier to classify emails as Spam or Not Spam.(4 Marks)

d) Predict whether an email with the following features is Spam or Not Spam:

 Contains_Offer = Yes

 Contains_Link = No

 Contains_Attachment = Yes
(2 Marks)

e) Evaluate the model using accuracy score and briefly comment on the result.
(1 Mark)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create the dataset
data = {
    'Contains_Offer': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Contains_Link': ['Yes', 'No', 'No', 'Yes', 'No', 'No'],
    'Contains_Attachment': ['Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Is_Spam': ['Spam', 'Not Spam', 'Spam', 'Not Spam', 'Spam', 'Not Spam']
}
df = pd.DataFrame(data)
Suitability of Naïve Bayes:

It is a probabilistic classifier that works efficiently with discrete, categorical, or binary features (Yes/No). It assumes feature independence, which, while "naive," makes it very fast to train and effective at calculating the probability of a document belonging to a class.

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical variables into numerical
df_encoded = df.copy()
for col in ['Contains_Offer', 'Contains_Link', 'Contains_Attachment', 'Is_Spam']:
    df_encoded[col] = le.fit_transform(df[col])

# Display encoded data (Yes=1, No=0 | Spam=1, Not Spam=0)
print(df_encoded)

from sklearn.naive_bayes import BernoulliNB
import numpy as np

# Features (X) and Target (y)
X = df_encoded[['Contains_Offer', 'Contains_Link', 'Contains_Attachment']]
y = df_encoded['Is_Spam']

# Initialize and train the model (BernoulliNB for binary data)
model = BernoulliNB()
model.fit(X, y)

print("Model trained successfully.")

# Predict for: Contains_Offer=Yes(1), Contains_Link=No(0), Contains_Attachment=Yes(1)
new_email = [[1, 0, 1]]
prediction = model.predict(new_email)
result = "Spam" if prediction[0] == 1 else "Not Spam"
print(f"Predicted Class: {result}")

# Evaluate
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy * 100:.2f}%")


5. A data analyst is given a dataset containing student academic and skill assessment data.
Due to the presence of multiple correlated features, Principal Component Analysis (PCA) is
used to reduce dimensionality retaining maximum information.

a) Write a Python program to standardize the dataset and explain why standardization is required before applying PCA. (3 Marks)

b) Apply PCA to reduce the dataset to two principal components and display the explained variance ratio. (4 Marks)

c) Create a 2D scatter plot using the two principal components and label the students. (3 Marks)


d) Using the two principal components, identify:
 Which student(s) show overall high performance, and
 Which student(s) show relatively low performance.
Justify your answer based on their position in the PCA scatter plot.
(2 Marks)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = {
    "Math": [78,65,88,55,92,70],
    "Physics": [72,60,85,58,90,68],
    "Chemistry": [75,62,82,60,88,72],
    "Programming": [85,70,90,65,95,75],
    "Aptitude": [80,68,86,62,90,74]
}

students = ["S1","S2","S3","S4","S5","S6"]
df = pd.DataFrame(data, index=students)
#a) Write a Python program to standardize the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

#b) Apply PCA to reduce the dataset to two principal components and display the explained variance ratio.
#(4 Marks)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

#c) Create a 2D scatter plot using the two principal components and label the students.
#(3 Marks)
plt.scatter(pca_data[:,0], pca_data[:,1])

for i in range(len(students)):
    plt.text(pca_data[i,0], pca_data[i,1], students[i])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter Plot of Students")
plt.show()

Explanation of Outputs



a) Standardization
StandardScaler was used to transform the data to mean=0 and standard deviation=1. This is crucial because if one score (e.g., Programming) had a significantly larger range, PCA would wrongly prioritize it, skewing the results.



b) PCA & Explained Variance
The program reduces the 5 features to 2 Principal Components (PC1 and PC2). The output pca.explained_variance_ratio_ shows how much information is retained.



Typically, PC1 represents overall magnitude (total performance), while PC2 might represent the difference between academic and skill scores.



c) 2D Scatter Plot
X-axis (PC1): High values on the right represent high overall performance.
Y-axis (PC2): Separates students based on variance not explained by PC1.


d) Performance Identification
High Performance: S5 sits furthest to the right on the PC1 axis, indicating they have the highest scores across all subjects.


Low Performance: S4 sits furthest to the left on the PC1 axis, indicating they have the lowest scores across subjects.


Justification: The first principal component (PC1) accounts for the largest share of variance. Since all scores are positively correlated, high PC1 values correspond directly to high overall performance.


#d) Using the two principal components, identify:
## Which student(s) show overall high performance, and
# Which student(s) show relatively low performance.

#d) Identify students with high and low performance.
# High performance typically aligns with higher values on the principal components that explain most of the variance.
# PC1 explains the most variance, so we sort by PC1 to find extremes.
df['PC1'] = pca_data[:, 0]
df['PC2'] = pca_data[:, 1]

#high_performers = df.sort_values(by='PC1', ascending=False)
low_performers = df.sort_values(by='PC1', ascending=True)

#high_performers
low_performers
