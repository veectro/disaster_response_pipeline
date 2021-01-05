import sys
from typing import Tuple, List

import re
import pandas as pd
import pickle

from pandas import DataFrame, Series
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score

URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath: str):
    """
    Load the SQLite db then split the dataset into data and target value.

    :param database_filepath: location of the SQLite DB
    :return: X as data value, Y as target value, category_names as list of target value
    """
    engine = create_engine(f'sqlite:///{database_filepath}')

    sql_query = "SELECT * FROM disaster_messages;"
    df = pd.read_sql(sql_query, con=engine)
    df.head()
    engine.dispose()

    X = df['message']  # data column
    Y = df.iloc[:, 4:]  # target column
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text: str, url_regex: str = URL_REGEX) -> List[str]:
    """
    Split given text into list of string. If  an url is exist, then replace with 'placeholder' (literally).
    For each string in the text, only the lemma (single items) will be returned. Ex : walked => walk

    :param text: text to be tokenized
    :param url_regex: regex for the URL. if exist, the default regex will be overriden
    :return: list of tokenized string
    """
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


def build_model() -> GridSearchCV:
    """
    Using GridSearchCV, the best parameter will be search for the chosen classifier

    :return: ML model
    """
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'text_pipeline__tfidf__use_idf': (True, False),
        'classifier__estimator__n_estimators': (50,150)
    }

    model = GridSearchCV(pipeline, n_jobs=-1, param_grid=parameters)

    return model


def evaluate_model(model: GridSearchCV, X_test: Series, Y_test: DataFrame, category_names: List[str]):
    """
    Evaluating the model by displaying its perfomance for each category, such as f1-score, accuracy.

    :param model: a trained model
    :param X_test: panda series containing test data
    :param Y_test: multi-label dataframe
    :param category_names: list of category from the target value
    :return: none
    """
    Y_pred = model.predict(X_test)

    # iterating for each target columns to calculate the f1-score, precision, and accuracy.
    for i in range(len(category_names)):
        print(f'----Classification Report for category :{category_names[i]}\n')
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print(f'Accuracy : {accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])}\n')
        print(f"F1-Score : {f1_score(Y_test.iloc[:, i].values, Y_pred[:, i], average='weighted')}\n")


def save_model(model: GridSearchCV, model_filepath: str):
    """
    Saving given ML Model as a pickle file in given filepath.

    :param model: a trained model
    :param model_filepath: location to save pickle file
    """
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
