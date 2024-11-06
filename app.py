from flask import Flask, request, render_template
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Sample data and model setup (same as before)
X = np.array([
    [139, 27.1, 57],
    [85, 22.0, 36],
    [116, 25.6, 54],
    [78, 31.0, 29],
    [115, 35.3, 46],
])
y = np.array([1, 0, 1, 0, 1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])
        
        input_scaled = scaler.transform([[glucose, bmi, age]])
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = "You may have diabetes" if prediction[0] == 1 else "You may not have diabetes"
        return render_template('result.html', result=result, probability=probability)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)