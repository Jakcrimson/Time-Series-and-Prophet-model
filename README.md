# Time-Series-and-Prophet-model
This repo contains an overview of how Long Short Term Memory (LSTM) models work on financial data from the Yahoo finance api. It also includes a short overview of the Facebook Prophet Model.

## Why 
This project has a teaching purpose, i've been interested in time series for a long time and financial data is one of the most common applications to start with.

## With what
I've been able to write the notebooks with the help of papers on Towards Data Science and Medium (for the Facebook Prophet model setup I refered directly to the repository's documentation).

## In order to .. ?
Get to know how time series work and maybe find other fields of application, visualizing data (geodata for example ;) ), comparing algorithms. But most of all I wanted to prepare myself for my internship this summer 2022 at PureControl in Rennes. I will probably be working on time series and time data analysis.

# Files description
All notebooks should have decent comments.

LSTM_MODEL.h5 -> the first file in the repository is the saved LSTM_model used in the application app.py (streamlit app not deployed yet).

LSTM_Stock_prediction_2.ipynb -> the jupyter notebook used to train, test and save the LSTM_MODEL.h5. It's using data from the Yahoo Finance API. You are welcome to change to stock ticker to any company you would like the model to be built upon (default is APPLE).

app.py -> the application, the code is not very well organized. You need stramlit to run it and if you do you have to run it from the folder of the repo with : streamlit run app.py

prophet_stock_pred.py -> jupyter notebook where i'm trying out the facebook prophet model and differen functions it has. ot actually commented, i'm still working on it.

requirements.txt -> the librairies you need to run the app and the notebooks.

ALL FILES RELATED TO STOCKER.PY / STOCKER.CPYTHON-36.PYC ARE TO BE IGNORED THEY WILL BE DELETED SOON.