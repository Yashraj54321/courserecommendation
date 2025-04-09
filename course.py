import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset and encoders
df_cleaned = pd.read_excel("cleaned_dataset.xlsx")


try:
    label_encoders = joblib.load("label_encoders.pkl")
except:
    st.error("❌ Error: Label encoders file not found. Please retrain the encoders.")
    st.stop()

# Handle missing values (replace NaN with "Unknown")
df_cleaned.fillna("Unknown", inplace=True)

# Streamlit UI
st.title("🎓 AI-Powered Course Recommender")
st.write("Find the best courses tailored to your profile!")

# User Inputs
interest = st.selectbox("Select Your Interest:", df_cleaned['Interest'].unique())
career_goal = st.selectbox("Select Your Career Goal:", df_cleaned['Career Goal'].unique())
skills_learned = st.selectbox("Select Your Skills:", df_cleaned['Skills Learned'].unique())

# Handle multiple skills (take the first skill if comma-separated)
skills_list = skills_learned.split(", ")
if skills_list:
    skills_learned = skills_list[0]  # Take the first recognized skill

# Recommendation Function
def recommend_courses(interest, career_goal, skills_learned, top_n=5):
    try:
        # Check for unseen labels
        if interest not in label_encoders['Interest'].classes_:
            st.warning(f"⚠️ Unrecognized Interest: {interest} (Using 'Unknown')")
            interest = "Unknown"

        if career_goal not in label_encoders['Career Goal'].classes_:
            st.warning(f"⚠️ Unrecognized Career Goal: {career_goal} (Using 'Unknown')")
            career_goal = "Unknown"

        if skills_learned not in label_encoders['Skills Learned'].classes_:
            st.warning(f"⚠️ Unrecognized Skill: {skills_learned} (Using 'Unknown')")
            skills_learned = "Unknown"

        # Encode user inputs
        user_profile = np.array([
            label_encoders['Interest'].transform([interest])[0],
            label_encoders['Career Goal'].transform([career_goal])[0],
            label_encoders['Skills Learned'].transform([skills_learned])[0],
        ]).reshape(1, -1)

        # Encode dataset features
        encoded_features = df_cleaned[['Interest', 'Career Goal', 'Skills Learned']].copy()
        for col in ['Interest', 'Career Goal', 'Skills Learned']:
            encoded_features[col] = label_encoders[col].transform(encoded_features[col])

        course_features = encoded_features.values

        # Compute similarity
        user_similarity = cosine_similarity(user_profile, course_features).flatten()

        # Get top N recommended courses
        top_indices = user_similarity.argsort()[::-1]
        recommended_courses = df_cleaned.iloc[top_indices][
            ['Course Title', 'Course Level', 'Course Rating']].drop_duplicates().head(top_n)

        if recommended_courses.empty:
            st.error("❌ No recommendations found. Try selecting different options.")
            return pd.DataFrame()

        return recommended_courses

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Show Recommendations on Button Click
if st.button("Get Recommendations"):
    st.write("🔍 Fetching recommendations...")  # Debugging message
    recommendations = recommend_courses(interest, career_goal, skills_learned)

    if recommendations.empty:
        st.write("❌ No recommendations found. Try selecting different options.")
    else:
        st.write("recommended courses are--")
        st.table(recommendations)