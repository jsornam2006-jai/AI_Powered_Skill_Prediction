import streamlit as st
import pandas as pd
import joblib

# Sidebar
with st.sidebar:
    st.header("📌 About the Project")

    st.write(
        "AI-Powered Skill Demand Prediction System"
    )

    st.write("### Technologies")
    st.write("🐍 Python")
    st.write("🤖 Scikit-learn")
    st.write("📊 Pandas")
    st.write("⚡ Streamlit")
    st.write("💾 Joblib")

    st.write("### ML Models")
    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• Random Forest")

    st.info(
        "The system predicts suitable career paths "
        "based on selected technical skills."
    )


# Load trained model
model_data = joblib.load("models/career_model.pkl")

model = model_data["model"]
features = model_data["features"]


# Page configuration
st.set_page_config(
    page_title="AI Powered Skill Prediction System",
    page_icon="🎯",
    layout="wide"
)


# Title
st.title("🎯 AI-Powered Skill Demand Prediction System")

st.write(
    "Select the skills you currently have and let the Machine Learning "
    "model predict the most suitable career path."
)

st.divider()


# Skill selection
st.subheader("🧠 Select Your Skills")

selected_skills = {}

for skill in features:
    selected_skills[skill] = st.checkbox(
        skill.replace("_", " "),
        key=skill
    )


# Prediction button
if st.button("🔮 Predict Career", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame(
        [[int(selected_skills[skill]) for skill in features]],
        columns=features
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"🎯 Predicted Career: **{prediction}**")

    # Prediction probability
    probabilities = model.predict_proba(input_data)[0]

    probability_data = pd.DataFrame({
        "Career": model.classes_,
        "Probability": probabilities * 100
    })

    probability_data = probability_data.sort_values(
        "Probability",
        ascending=False
    )

    st.subheader("📊 Career Prediction Probability")

    st.bar_chart(
        probability_data.set_index("Career")["Probability"]
    )

    # Display probability table
    st.dataframe(
        probability_data.style.format(
            {"Probability": "{:.2f}%"}
        ),
        use_container_width=True
    )

    # Learning Recommendations
    st.subheader("📚 Learning Recommendations")

    learning_paths = {
        "Machine Learning Engineer": [
            "Advanced Python",
            "Machine Learning",
            "Deep Learning",
            "SQL",
            "Cloud Computing"
        ],
        "AI Engineer": [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "Natural Language Processing",
            "Cloud Computing"
        ],
        "Data Scientist": [
            "Python",
            "SQL",
            "Machine Learning",
            "Data Analysis",
            "Natural Language Processing"
        ],
        "Data Analyst": [
            "Python",
            "SQL",
            "Data Analysis",
            "Power BI"
        ],
        "Web Developer": [
            "Python",
            "Web Development",
            "SQL",
            "JavaScript"
        ],
        "Software Developer": [
            "Python",
            "Java",
            "SQL",
            "Web Development"
        ]
    }

    recommended_learning = learning_paths.get(prediction, [])

    st.write(
        f"Based on your predicted career **{prediction}**, "
        "you can focus on learning:"
    )

    for skill in recommended_learning:
        st.write(f"📖 **{skill}**")

        
    # Career Recommendations
    st.subheader("💡 Career Recommendations")

    recommended_careers = probability_data[
        probability_data["Probability"] >= 10
    ]

    if len(recommended_careers) > 1:
        st.write("Based on your selected skills, you may also consider:")

        for _, row in recommended_careers.iloc[1:].iterrows():
            st.write(
                f"🔹 **{row['Career']}** — "
                f"{row['Probability']:.2f}%"
            )

    else:
        st.write(
            "Your predicted career is the strongest match "
            "for the selected skills."
        )
    


st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn, Joblib and Streamlit."
)

# Skill Demand Analysis
st.divider()

st.subheader("📊 Skill Demand Analysis")

dataset = pd.read_csv("dataset/career_prediction.csv")

skill_demand = {}

for skill in features:
    demand = dataset[skill].sum() / len(dataset) * 100
    skill_demand[skill.replace("_", " ")] = demand

demand_data = pd.DataFrame(
    list(skill_demand.items()),
    columns=["Skill", "Demand"]
)

demand_data = demand_data.sort_values(
    "Demand",
    ascending=False
)

st.bar_chart(
    demand_data.set_index("Skill")["Demand"]
)

st.write("Skill demand is calculated from the project dataset.")

