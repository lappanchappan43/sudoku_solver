from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('tempalte/index.html')

@app.route('/solve')
def solve():
    pass

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=True)