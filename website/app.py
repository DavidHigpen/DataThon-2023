import csv 
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, jsonify
# from app.routes import configure_routes
# from matplotlib import font_manager as fm
# from PIL import Image



data_list = []
app = Flask(__name__)
# configure_routes(app)

with open('StudentsPerformance_with_headers.csv') as csvfile: 
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data_list.append(row)

values = []
categories = []
    
with open('categories.txt', 'r') as file:
    values = [line.strip() for line in file.readlines()]
    for line in values:
        number = line[0 : line.index('-')]
        label = line[line.index('-') + 2 : line.index(':')]
        key = line[line.rfind('(') + 1 : -1]
        categories.append([number, label, key])

@app.route('/')
def index():
    return render_template('index.html')
    
# @app.route('/home')
# def index():
#     return render_template('index.html')
    
@app.route('/uploadForm') #I think something is wrong here
def uploadForm():
    return render_template('uploadForm.html')

@app.route('/process', methods=['POST'])
def process():
    # Get input from the POST request
    input_data = request.get_json()

    # Process the input data (parse comma-separated values)
    # data_values = list(map(float, input_data['data'].split(',')))
    category = input_data['data']
    if(category.isdigit()): i = int(category)
    else: i = data_list[0].index(category)
    # print("OUTPUT DATA:", category)

    data = []
    for row in data_list[1:]:
        data.append(int(row[i]))
        
    # custom_font = fm.FontProperties(family='Josefin Sans', size=12)

    fig = plt.figure(facecolor='#ebded5')
    plt.hist(data, bins=range(min(data), max(data) + 2), align='left', rwidth=0.8, edgecolor='black', color='#899289')
    plt.gca().set_facecolor('#ebded5')
    plt.xticks(range(min(data), max(data) + 1))
    plt.title(categories[i - 1][1], color='#af7547')
    plt.xlabel(categories[i - 1][2], color='#af7547')
    plt.ylabel('Number of datapoints', color='#af7547')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # resizing_factor = 0.6
    # img = Image.open(img)
    # img = img.resize((int(img.width * resizing_factor), int(img.height * resizing_factor)))
    
    # Convert the BytesIO object to base64 for HTML embedding
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Close the plot to free up resources
    plt.close()
    average = 0
    
    for row in data_list[1:]:
        average += int(row[i])
    average /= len(data_list)
    output = formatted_number = "{:.2f}".format(average)
    
    # Return the output and the base64-encoded plot
    output_data = "Python processed data: "
    return jsonify({'plot': plot_base64, 'output': output})

if __name__ == '__main__':
    app.run(debug=True)
