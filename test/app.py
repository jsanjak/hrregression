from flask import Flask, render_template
import pandas as pd

df = pd.read_csv('https://data.boston.gov/dataset/c8b8ef8c-dd31-4e4e-bf19-af7e4e0d7f36/resource/29e74884-a777-4242-9fcc-c30aaaf3fb10/download/economic-indicators.csv',
                 parse_dates=[['Year', 'Month']])
length = len(df)
#always instantiate a flask object with the following boilerplate code
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html', length=length,
                           dataframe=df.to_html(classes='table'))


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5000)
