from  flask import Flask, render_template, request, jsonify
from mainish import *



def square(x):
    return x*x

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_data = request.form['input_data']
    output_data = square(int(input_data))
    return jsonify(output_data=output_data)

if __name__ == '__main__':
    app.run(debug=True)