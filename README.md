# Disaster Response Pipeline Project

## Dependencies: 
This project used `python:3.8` and following dependencies: 
- flask:1.1.2
- nltk:3.5
- pandas:1.1.5
- plotly:4.14.1
- scikit-learn:0.23.2
- sqlalchemy:1.3.21

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
