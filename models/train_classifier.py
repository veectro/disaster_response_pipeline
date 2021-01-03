import sys
import numpy as np
import re
import pandas as pd
import pickle

from pandas import DataFrame
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, precision_score, recall_score, \
    f1_score, accuracy_score

URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath: str) -> DataFrame:
    engine = create_engine(f'sqlite:///{database_filepath}')

    sql_query = "SELECT * FROM disaster_messages;"
    df = pd.read_sql(sql_query, con=engine)
    df.head()
    engine.dispose()

    X = df['message']  # feature column
    Y = df.iloc[:, -36:]  # label column
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text: str, url_regex: str = URL_REGEX):
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'placeholder')

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('rfc_classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'text_pipeline__tfidf__use_idf': (True, False),
        'rfc_classifier__estimator__n_estimators': [10, 20]
    }

    model = GridSearchCV(pipeline, n_jobs=-1, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    #TODO


def save_model(model, model_filepath):
    # create a pickle file for the model
    file_name = model_filepath
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
