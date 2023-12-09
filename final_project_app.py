import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

s = pd.read_csv("social_media_usage_2.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]).astype(int),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]).astype(int),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"]).astype(int)
})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 987)

lr = LogisticRegression(class_weight = "balanced", random_state = 987)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)


# STREAMLIT APP --------------------------------------------------------------------------------------

st.write("""
         
# Are YOU a LinkedIn User?
         
## Let's predict whether or not you use LinkedIn.
         
### Tell me about yourself:

""")

# Income
income_options = {
    "Less than $10k": 1,
    "$10k to under $20k": 2,
    "$20k to under $30k": 3,
    "$30k to under $40k": 4,
    "$40k to under $50k": 5,
    "$50k to under $75k": 6,
    "$75k to under $100k": 7,
    "$100k to under $150k": 8,
    "$150k or more": 9
}

income_key = st.selectbox("Income Level", list(income_options.keys()))
income_value = income_options[income_key]

# Education
education_options = {
    "Less than high school (Grades 1-8 or no formal schooling)": 1,
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)": 2,
    "High school graduate (Grade 12 with diploma or GED certificate)": 3,
    "Some college, no degree (includes some community college)": 4,
    "Two-year associate degree from a college or university": 5,
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)": 6,
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)": 7,
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)": 8
}

education_key = st.selectbox("Education Level", list(education_options.keys()))
education_value = education_options[education_key]

# Parent
parent_options = {
    "Yes": 1,
    "No": 0
}

parent_key = st.selectbox("Parent", list(parent_options.keys()))
parent_value = parent_options[parent_key]

# Married
married_options = {
    "Yes": 1,
    "No": 0
}

married_key = st.selectbox("Married", list(married_options.keys()))
married_value = married_options[married_key]

# Gender
gender_options = {
    "Male": 0,
    "Female": 1
}

gender_key = st.selectbox("Gender", list(gender_options.keys()))
gender_value = gender_options[gender_key]

# Age
age_value = st.number_input("Age", min_value = 0, max_value = 98, value = 10, step = 1)

user_input = pd.DataFrame({
    "income": [income_value],
    "education": [education_value],
    "parent": [parent_value],
    "married": [married_value],
    "female": [gender_value],
    "age": [age_value]
})

user_result = lr.predict(user_input)
user_prob = lr.predict_proba(user_input)

if user_result == 1:
    st.markdown("### Yes! You probably use LinkedIn.")
else:
    st.markdown("### No, you probably don't use LinkedIn.")

st.markdown(f"#### Probability of Using LinkedIn: {user_prob[0][1]}")