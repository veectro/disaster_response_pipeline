import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from pandas import DataFrame
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def calculate_target_frequency(df: DataFrame) -> DataFrame:
    """
    Generate a new dataframe that will calculate the frequency of the category value

    :param df: dataset from the db
    :return: a dataframe with 2 columns, category and frequency
    """
    Y = df.iloc[:, -36:]  # target column

    cats = []
    frequency = []
    for col in Y.columns:
        cats.append(col[9:])
        frequency.append(Y[col][Y[col] == True].sum())

    return pd.DataFrame({'category': cats, 'frequency': frequency})


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    genre_graph = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }


    # get category-frequency data
    freq_df = calculate_target_frequency(df).sort_values(by='frequency', ascending=False)
    freq_graph = {
            'data': [
                Bar(
                    x=freq_df['category'],
                    y=freq_df['frequency']
                )
            ],

            'layout': {
                'title': 'Distribution of Category',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }

    # create visuals
    graphs = [
        genre_graph,
        freq_graph
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # get first 100 messages
    messages = list(df['message'].head(100))

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, messages=messages)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(port=3001, debug=True)


if __name__ == '__main__':
    main()
