from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

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

# DataFrame to store tested values
tested_values = pd.DataFrame(columns=["Username", "Height", "Predicted Weight"])

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_credentials(username, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        height = float(request.form['height'])
        prediction = model.predict([[height]])
        predicted_weight = float(prediction[0])
        
        # Store the tested value
        global tested_values
        # tested_values = tested_values.add({
        #     "Username": session['username'],
        #     "Height": height,
        #     "Predicted Weight": predicted_weight
        # }, ignore_index=True)
        
        return render_template('index.html', predicted_weight=predicted_weight)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

