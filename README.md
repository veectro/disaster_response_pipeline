# Disaster Response Pipeline Project
This project is a part of Udacity Data Scientist Nano degree program.

## **Table of Contents:**
1. [Project Introduction](README.md#project-introduction)
2. [File Description](README.md#file-description)
3. [Libraries used](README.md#libraries-used)
4. [Results](README.md#results)


## Project Introduction
In this project, I will apply the data engineering skill to analyze disaster data from [Figure Eight](https://www.figure-eight.com/)
to build a machine learning (ML) model for an API that classifies disaster messages.

---

## **File Description**
    .
    ├── app                             # flask web app
    |   └── templates/                  # flask templates (html)
    |   └── run.py                      # flask main method, begin here
    ├── data                            # dataset and etl pipeline
    |   └── disaster_categories.csv     # categories dataset (multi-label)
    |   └── disaster_messages.csv       # messages dataset  
    |   └── DisasterResponse.db         # sqlite db created from the pipeline  
    |   └── process_data.py             # an etl pipeline to generate DisasterResponse.db from dataset
    ├── img                             # image folder for README.md
    ├── models                          # model related folder
    |   └── classifier.pkl              # ml model in pickle file  
    |   └── train_classifier.py         # an ML pipeline to train ML model from the dataset
    ├── notebooks                       # Jupyter Notebook that contains the analysis
    ├── README.md               
    └── requirements.txt                # dependency library

The dataset is originated from [insideairbnb](http://data.insideairbnb.com/germany/bv/munich/2020-10-26/data/listings.csv.gz) from the 26th October 2020. This dataset is licensed under [Creative Commons CC0 1.0 Universal (CC0 1.0) "Public Domain Dedication"](https://creativecommons.org/publicdomain/zero/1.0/)

---
## Methodology
1. ETL Pipeline 
   In this step, I will load the given data, merge the dataset, process it by cleaning duplicate rows, cleaning and splitting a column into 
   multi-label column, then save it in SQLite-DB.
  
2. ML Pipeline 
   From the previous step, I will load the database, and I will train it with `scikit-learn` library. 
   Multiple ML Algorithm will be trained too, and from the best algorithm, I will find the best parameter. 
   In the end, only the best model with the best parameter will be exported into the final ML model.
   
3. Flask Web App 
   The ML Model created from the pipeline before will be loaded into a `flask` web app, so other user could test it.

---

## Libraries used: 
The dependencies could be installed with command: 
```python
pip install requirements.txt 
```

This project used `python:3.8` and following dependencies: 
- flask:1.1.2
- nltk:3.5
- pandas:1.1.5
- plotly:4.14.1
- scikit-learn:0.23.2
- sqlalchemy:1.3.21

### Instructions to generate ML model and run the `web app`:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    ```python
    cd app
    python run.py
    ```

3. Go to http://127.0.0.1:3001/

---

## Results 
### Main Menu
![Main Page](img/main.png?raw=true "Main Menu")
### Result
![Classify Page](img/result.png?raw=true)


---

## Acknowledgements
Credit and thanks too:

- Udacity for providing a great data scientist course.
- Figure Eight to providing a great data.