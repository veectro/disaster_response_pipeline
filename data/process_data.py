import sys
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    """
    Load the dataset as pandas dataframe with given paths.
    The dataset will be then merged by left join on id.

    :param messages_filepath: filepath to message.csv file
    :param categories_filepath: filepath to category.csv file
    :return: pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df: DataFrame) -> DataFrame:
    """
    Process the dataframe by replacing the `categories` column with 36 new columns containing
    each category.

    - process the `categories` column
    - replace `categories` column with the 36 new created columns
    - dropping

    :param df: dataframe that need to be cleaned
    :return: cleaned pandas DataFrame
    """
    categories = prepare_categories_column(df)

    # replacing the `categories` column with the new splitted columns
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates of row
    df.drop_duplicates(inplace=True)
    return df


def prepare_categories_column(df: DataFrame) -> DataFrame:
    """
    Clean the dataframe by splitting the `categories` column into 36 new columns
    then rename the last 36 columns (category) with the category name prefixed with
    `category:`.

    :param df: the original dataframe containing `categories`
    :return: a panda dataframe containing 36 `categories` as each column
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[-1]

    prefix = 'category:'
    # extract a list of new column names for categories.
    # ex : related-1 => category:related
    category_colnames = row.apply(lambda col_name: prefix + col_name[:-2]).values
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda col_name: col_name[-1:]).values

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    return categories


def save_data(df: DataFrame, database_filename: str):
    """
    Save given dataframe into a sqllite db.

    :param df: dataframe to be saved
    :param database_filename: location of the sqllite
    :return:
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', if_exists='replace', con=engine, index=False)
    engine.dispose()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
