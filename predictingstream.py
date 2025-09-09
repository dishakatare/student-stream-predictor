import streamlit as st
import numpy as np
import pickle
import time

# ✅ Load Model
model_path = "streampredict.pkl"   # make sure you have trained & saved model as stream.pkl
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found! Please train and save as stream.pkl")
    st.stop()

# ✅ Themed UI Styling
st.set_page_config(page_title="Student Stream Predictor", page_icon="🎓", layout="wide")
st.markdown("""
    <style>
    body {background-color: #f0f9ff; color: #2c3e50; font-family: 'Poppins', sans-serif;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 20px; font-size: 18px; padding: 10px 22px;}
    .title {font-size: 48px; font-weight: bold; color: #0077b6; text-align: center;}
    .subtitle {font-size: 22px; color: #023e8a; text-align: center; font-style: italic;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>🎓 Student Stream Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>📘 Suggests Science, Commerce, or Arts based on marks & interests</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ About the App")
    st.write("This app predicts which stream (Science / Commerce / Arts) a student should choose after 10th standard.")
    st.write("It uses a trained Machine Learning model (Decision Tree/Random Forest).")
    st.header("💡 Tips")
    st.write("✔ Enter marks honestly for better predictions.")
    st.write("✔ Use the results as guidance, not as the final decision.")

# ✅ Input Fields
st.subheader("📚 Enter Your 10th Marks")
maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=70)
science = st.number_input("Science Marks", min_value=0, max_value=100, value=70)
english = st.number_input("English Marks", min_value=0, max_value=100, value=70)
social = st.number_input("Social Science Marks", min_value=0, max_value=100, value=70)
language = st.number_input("Language Marks", min_value=0, max_value=100, value=70)

st.subheader("📌 Interest Survey (1 = Low, 5 = High)")
interest_math = st.slider("Interest in Maths", 1, 5, 3)
interest_science = st.slider("Interest in Science", 1, 5, 3)
interest_business = st.slider("Interest in Business/Commerce", 1, 5, 3)
interest_arts = st.slider("Interest in Arts/Humanities", 1, 5, 3)

st.subheader("🧠 Skills (1 = Low, 5 = High)")
analytical = st.slider("Analytical Skill", 1, 5, 3)
creativity = st.slider("Creativity Skill", 1, 5, 3)
communication = st.slider("Communication Skill", 1, 5, 3)
problem_solving = st.slider("Problem Solving Skill", 1, 5, 3)



# Define mapping
stream_mapping = {0: "SCIENCE", 1: "COMMERCE", 2: "ARTS"}

# ✅ Prediction Logic
if st.button("📊 Predict My Stream"):
    with st.spinner('⏳ Analyzing your marks and interests...'):
        time.sleep(2)
        input_features = [
            maths, science, english, social, language,
            interest_math, interest_science, interest_business, interest_arts,
            analytical, creativity, communication, problem_solving
        ]
        try:
            pred_int = model.predict(np.array(input_features).reshape(1, -1))[0]
            stream_label = stream_mapping.get(pred_int, "Unknown")

            st.success(f"✅ Recommended Stream: **{stream_label}**")

            # Optional: show career suggestions
            if stream_label == "SCIENCE":
                st.info("🔬 Possible Careers: Engineer, Doctor, Researcher, Scientist, IT Professional")
            elif stream_label == "COMMERCE":
                st.info("💹 Possible Careers: CA, Accountant, MBA, Banker, Entrepreneur")
            else:
                st.info("🎨 Possible Careers: Writer, Artist, Teacher, Journalist, Psychologist")

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

