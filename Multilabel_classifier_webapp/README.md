# MULTI-LABEL INTENT CLASSIFIER APP PROJECT
The multi-label intent classifier model is built using **XLNet** and the app is created using **Streamlit**.  
MAY'21-JULY 21

# Demo
This is the preview of the app with an example text already tried out.

![alt text](demo1.png?raw=true)

You can check the other demo images of the app given in the repository.

# Overview
A Multi-label intent classifier Project which predicts the sentiment and multiple intents of
the review of a google play store app along with the percentage of positivity/negativity and also the percentage of
each label.

# Technical Aspect
This project is divided into two parts:
- Building the classifier model:  
https://github.com/rishikavaish/Projects-ML/blob/a1a46a573f6cd9235ba75fe9a83f2c525348536b/Multilabel_classifier_webapp/building_the_classifier_model.ipynb
- Applying the classifier model to create multi-label intent classifier web app using streamlit:  
https://github.com/rishikavaish/Projects-ML/blob/7ea935e90c8d6c5885f7025258fe59d2ec32add8/Multilabel_classifier_webapp/multilabel_classifier_app.py

# Files required
- Clusterd Data CSV: https://drive.google.com/file/d/1I_QKBMTsEWMYmLmMD6chu1g9XDL0uSLH/view?usp=sharing  
- Trained model: https://drive.google.com/file/d/1Nh6KWiUnjgJk07xIRWxHI21G-17eW5Cr/view?usp=sharing

# Installation
The Code is written in Python 3.7. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

```
pip install -r requirements.txt
```

# Steps
- Build the Classifier model using NLP
  - Scrape reviews from a google playstore app
  - Text Preprocessing
  - Text clustering
  - Fine tune a pretrained XLNet model
  - Generate Predictions
- Applying the classifier model to create a multilabel intent classifier app using Streamlit

# Summary

- First, the reviews of the google play store app are scraped and formed into a dataset accordingly with various other columns like date and time of the review, star rating, thumbs up count, etc., which will be used in the exploratory data analysis. And these reviews along with other information are scraped using the Google Play scraper and saved batch wise using MongoDB.
- The text of the reviews is preprocessed for text clustering by lower casing the text, removing punctuations, lemmatization, stemming and tokenization using NLTK library. 
- The next step is the sentiment analysis i.e. the sentiment classification of the
reviews into positive, negative and neutral along with the percentage of positivity or negativity.This sentiment classification is done using the VADER sentiment and then the percentage of each sentiment is also calculated. It is a pretrained model which takes the input from the text description and outputs the sentiment score ranging from -1 to +1 for each sentence. VADER not only tells about the Positive and Negative score but also tells us about how positive or negative a sentiment is. Some features of VADER:
   - Emoticons (e.g. ! has value)
   - Degree modifiers ('extremely good' vs 'sort of good')
   - Shift in polarity due to but (e.g. I liked the app, but the speed was slow.)
- Now to get to know more about the problems that the users are facing, the next step is the intent analysis, i.e. to classify the text into different categories, something other than the basic positive, negative and neutral classification.
- For intent classification, first text clustering was done using tf-idf and then k-means clustering was applied on it to form the clusters. Then, there is a n-grams technique which is used to clean the clusters. In this n-grams technique, you have to make n-grams for each cluster and pick only those words which belong to that particular category, basically, the words which are more frequent in that cluster. So, this was how the final clusters were formed. The following seven categories were formed after clustering.
```
label_cols = ['Problem in recharge','Problem in reward/redeem points','Problem in registration/login/username/password','Problem with customer care service','Other complaints','Bad/Irrelevant comments','Appreciation']
```
- After clustering, results were formed into a dataset and after doing some data wrangling, a final clustered dataset was formed where each review could have multiple labels. Below is the csv file for the complete clustered data which will be used afterwards and in this you can also see the format in which you have to feed the data to the classifier model:
   - https://drive.google.com/file/d/1I_QKBMTsEWMYmLmMD6chu1g9XDL0uSLH/view?usp=sharing
- So now comes the part to build a classifier model which will predict the category of a new review or a new set of reviews given to it. A pretrained XLNet model has been used for building the classifier and then this pretrained XLNet model was fine tuned by training it on our type of data. And, after preprocessing the data and training the model on this data, a function was formed for getting predictions for a new data along with the probabilities of each label.
- Here, I have attached the link of the trained model which has already been trained
on the clustered data csv:
   - https://drive.google.com/file/d/1Nh6KWiUnjgJk07xIRWxHI21G-17eW5Cr/view?usp=sharing
- Now comes the part to predict these labels for a single text as well. So for that, a web application was made to try out the classifier, first the labels to be predicted are displayed, then the user can enter a text and get the sentiment and multiple intents of the text.

## Model Execution Time:
- It takes around 45mins - 1 hour to train the model from scratch and then 45 mins - 1 hour more to train the model from the previous checkpoint.
- It takes 1s-20s to generate a single prediction using this model.

## Technologies/libraries used
- Python
- google-play-scraper
- MongoDB
- NLTK
- TF-IDF
- K-Means
- Keras
- Tensorflow
- Torch
- XLNet
- Transformers
- VADER Sentiment
- Streamlit

## Made by:
RISHIKA VAISH  
Email id: rishikavaish321@gmail.com
