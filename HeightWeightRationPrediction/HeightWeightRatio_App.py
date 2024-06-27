import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_path = "C:/sudhanshu_projects/project-task-training-course/HeightWeightRatio.pkl"
model = pickle.load(open(model_path, "rb"))

# User credentials DataFrame
credentials = {
    "username": ["user1", "user2"],
    "password": ["pass1", "pass2"]
}
credentials_df = pd.DataFrame(credentials)

# Function to check credentials
def check_credentials(username, password):
    if username in credentials_df["username"].values:
        user_index = credentials_df[credentials_df["username"] == username].index[0]
        if credentials_df.at[user_index, "password"] == password:
            return True
    return False

# Streamlit application
def main():
    st.title("Height to Weight Prediction")

    menu = ["Login", "Main Page"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_credentials(username, password):
                st.success("Logged in successfully")
                st.experimental_set_query_params(page="main")
            else:
                st.error("Invalid username or password")

    elif choice == "Main Page":
        st.subheader("Predict Weight from Height")

        height = st.number_input("Enter height in inches", min_value=0.0, step=0.1)

        if st.button("Predict"):
            if height > 0:
                prediction = model.predict([[height]])
                predicted_weight = float(prediction[0])
                st.success(f"The predicted weight is {predicted_weight:.2f} pounds")
            else:
                st.error("Please enter a valid height")

if __name__ == "__main__":
    main()
