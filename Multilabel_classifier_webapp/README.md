# MULTI-LABEL INTENT CLASSIFIER PROJECT
The multi-label intent classifier model is built using **XLNet** and the app is created using **Streamlit**.  
MAY'21-JULY 21

Overview
A Multi-label intent classifier Project which predicts the sentiment and multiple intents of
the given text along with the percentage of positivity/negativity and also the percentage of
each label.
Summary
● Firstly, the reviews of the SBI rewardz app are scraped from 1st Jan’19 to 31st
Mar’21 (2 years 3 months) and formed into a dataset accordingly with various other
columns like date and time of the review, star rating, thumbs up count, etc., which
will be used in the analysis. And these reviews along with other information are
scraped using the Google Play scraper and saved batch wise using MongoDB.
● The text of the reviews is preprocessed for sentiment analysis(using VADER
sentiment) by lower casing the text, removing punctuations, lemmatization and
stemming.TextBlob Library was used for text preprocessing.TextBlob is built upon
NLTK and provides an easy to use interface to the NLTK library. Stopwords removal
does not yield better results because VADER uses words such as 'but' in calculating
the compound score. Therefore, stopwords are not removed from the text.
● To get to know more about the dataset, Exploratory data analysis was performed on
the dataset. Exploratory data analysis gave results like the number of reviews for
each score, text word count for each score, reviews for each version and other
yearly and quarter on quarter trends.
● The next step is the sentiment analysis i.e. the sentiment classification of the
reviews into positive, negative and neutral along with the percentage of positivity or

1

negativity.This sentiment classification is done using the VADER sentiment and then
the percentage of each sentiment is also calculated. It is a pretrained model which
takes the input from the text description and outputs the sentiment score ranging
from -1 to +1 for each sentence. VADER not only tells about the Positive and
Negative score but also tells us about how positive or negative a sentiment is. Some
features of VADER:
○ Emoticons (e.g. ! has value)
○ Degree modifiers ('extremely good' vs 'sort of good')
○ Shift in polarity due to but (e.g. I liked the app, but the speed was slow.)
● To get to know more about these positive, negative and neutral reviews, some
sentiment analysis results were found like the number of positive, negative and
neutral reviews, word cloud for positive and negative reviews and other yearly
and quarterly trends of positive and negative reviews.
● Now to get to know more about the problems that the users are facing, the Next
step is the intent analysis, i.e. to classify the text into different categories,
something other than the basic positive, negative and neutral classification.
● For intent classification, first text clustering was done using tf-idf and then
k-means clustering was applied on it to form the clusters. Then, there is a
n-grams technique which is used to clean the clusters. In this n-grams technique,
you have to make n-grams for each cluster and pick only those words which
belong to that particular category, basically, the words which are more frequent in
that cluster. So, this was how the final clusters were formed. Seven categories
were formed after clustering.
○ label_cols = ['Problem in recharge','Problem in reward/redeem
points','Problem in
registration/login/username/password','Problem with customer
care service','Other complaints','Bad/Irrelevant
comments','Appreciation']

● After clustering, results were formed into a dataset and after doing some data
wrangling, a final clustered dataset was formed where each review could have

2

multiple labels. Below is the csv file for the complete clustered data which will be
used afterwards:
○ https://drive.google.com/file/d/1I_QKBMTsEWMYmLmMD6chu1g9XDL0u
SLH/view?usp=sharing

● Then the count of each category and the bar graph for the same is printed. Then
a bar plot is built which displays how many reviews have multiple categories. So
if a review has more than 2 categories, then also it will be displayed here and also
if it doesn't belong to any category.
● So now comes the part to build a classifier model which will predict the category
of a new review or a new set of reviews given to it. A pretrained XLNet model has
been used for building the classifier and then this pretrained XLNet model was
fine tuned by training it on our type of data. And, after preprocessing the data and
training the model on this data, a function was formed for getting predictions for
a new data along with the probabilities of each label.
● Here, I have attached the link of the trained model which has already been trained
on the clustered data csv:
○ https://drive.google.com/file/d/1Nh6KWiUnjgJk07xIRWxHI21G-17eW5Cr/
view?usp=sharing

● The reviews which do not belong to any category are also printed. The reason to
display this information is, let's say if in future there are some reviews which do
not belong to category, so we will get to know how many such reviews there are.
So let's say if someday a lot of people start coming with the same problem, we
will get to know that it's time to form a new category. This ensures that our model
will not fail if some new problems arise in future.
● Now comes the part to predict these labels for a single text as well. So for that, a
web application was made to try out the classifier, first the labels to be predicted
are displayed, then the user can enter a text and get the sentiment and multiple
intents of the text.

3

## Model Execution Time:
- It takes around 45mins - 1 hour to train the model from scratch and then 45 mins - 1
hour more to train the model from the previous checkpoint.
- It takes 1s-20s to generate a single prediction using this model.

## Made by:
RISHIKA VAISH
Email id: rishikavaish321@gmail.com
