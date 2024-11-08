from flask import Flask, request, render_template, jsonify
import fruit_model  # Import your model training/testing script here

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # This should be the HTML file you’re using in Visual Studio Code

@app.route('/train', methods=['POST'])
def train_model():
    # Call your model training function here
    fruit_model.train()
    return jsonify({'status': 'Model training complete!'})

@app.route('/test', methods=['POST'])
def test_model():
    # Call your model testing function here and retrieve results
    results = fruit_model.test()
    return jsonify({'status': 'Model testing complete!', 'results': results})

if __name__ == '__main__':
    app.run(debug=True)
