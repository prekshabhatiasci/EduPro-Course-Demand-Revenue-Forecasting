# EduPro – Course Demand & Revenue Forecasting

## 📌 Project Overview

EduPro is an online learning platform that offers multiple courses across different categories, levels, and course types.

The objective of this project is to develop a **predictive analytics and forecasting system** that helps EduPro estimate future course enrollments and revenue.

The project uses historical transaction data, course information, teacher details, and time-based features to forecast **next-day course demand and revenue**.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Forecast next-day course enrollment.
- Forecast next-day course revenue.
- Identify trends in course demand over time.
- Analyze revenue performance across course categories.
- Compare different machine learning regression models.
- Identify important factors affecting enrollment and revenue.
- Provide an interactive Streamlit dashboard for business insights.
- Support data-driven decisions related to course launches, pricing, and instructor planning.

---

## 📊 Dataset

The project is based on the **EduPro Online Platform dataset** containing the following data:

### Users
- UserID
- UserName
- Age
- Gender
- Email

### Teachers
- TeacherID
- TeacherName
- Age
- Gender
- Expertise
- YearsOfExperience
- TeacherRating

### Courses
- CourseID
- CourseName
- CourseCategory
- CourseType
- CourseLevel
- CoursePrice
- CourseDuration
- CourseRating

### Transactions
- TransactionID
- UserID
- CourseID
- TransactionDate
- Amount
- PaymentMethod
- TeacherID

---

## 🔧 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Plotly**
- **Joblib**
- **Streamlit**
- **Jupyter Notebook**
- **VS Code**

---

## 🤖 Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor

The models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

> R² Score is used as a model evaluation metric and should not be interpreted as prediction accuracy percentage.

---

## 🧠 Feature Engineering

Time-based and historical features were created to improve forecasting.

Important features include:

- Previous-day enrollment
- Previous-day revenue
- Lag features
- Rolling averages
- Day of week
- Month
- Quarter
- Weekend indicator
- Course price
- Course duration
- Course rating
- Teacher experience
- Teacher rating
- Course category
- Course level
- Course type

The forecasting setup uses historical information to predict **future next-day values**, reducing data leakage.

---

## 📈 Forecasting Approach

The project follows a time-based forecasting workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Data Integration
     ↓
Feature Engineering
     ↓
Time-Based Features
     ↓
Lag & Rolling Features
     ↓
Train-Test Split
     ↓
Machine Learning Models
     ↓
Model Evaluation
     ↓
Next-Day Forecasting
     ↓
Streamlit Dashboard
