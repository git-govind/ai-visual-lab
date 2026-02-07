import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.title("🎓 Capstone: Build an AI System")

    model = st.selectbox(
        "Choose a model",
        ["Linear Regression", "Neural Network"]
    )

    noise = st.slider("Data Noise", 0.0, 5.0, 1.0)

    X = np.linspace(0, 10, 50)
    y = 2 * X + 3 + np.random.randn(50) * noise

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data")

    if model == "Linear Regression":
        w, b = np.polyfit(X, y, 1)
        ax.plot(X, w * X + b, color="red", label="Model")

    ax.legend()
    st.pyplot(fig)

    st.success(
        "You designed, trained, and evaluated an AI system."
    )

    if st.button("🎉 Finish Course"):
        st.session_state.completed.add("capstone")
