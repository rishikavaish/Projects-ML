# import required libraries
import warnings
warnings.filterwarnings("ignore")

from textblob import Word
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from wordcloud import WordCloud, STOPWORDS
from google_play_scraper import Sort, reviews
from datetime import datetime
from datetime import timedelta
import streamlit as st

# To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
import os
import math
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from transformers import XLNetTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

# Viz Pkgs
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')
sns.set_style('darkgrid')
STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

# contents in the sidebar
st.sidebar.header("About App")
st.sidebar.info("A SBI Rewardz App reviews analysis Project which will scrape reviews for the past 7 days. The extracted reviews will then be used to determine the Sentiments and Intents of those reviews. \
                    The different Visualizations will help us get a feel of the overall mood of the users regarding the App.")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("rishikavaish321@gmail.com")

# # building the structure of the application:

#heading
html_temp = """
<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live SBI Rewardz App Reviews Analysis</p></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.subheader("Results for past 7 days")

# code to extract reviews for past 1 week from a google play store app using google-play-scraper
result, continuation_token = reviews(
    'com.freedomrewardz',  # found in app's url
    lang='en',  # defaults to 'en'
    country='us',  # defaults to 'us'
    sort=Sort.NEWEST,  # start with most recent
    count=500  # batch size
)
df = pd.DataFrame(result)
df['at'] = pd.to_datetime(df['at'])
start_date = datetime.today() - timedelta(days=7)
end_date = datetime.today()
mask = (df['at'] > start_date) & (df['at'] <= end_date)
df = df.loc[mask]
df.drop(['repliedAt', 'userImage'], axis='columns', inplace=True)
x = df['reviewId'].count()
# Show the dimension of the dataframe
st.subheader("Number of rows and columns")
st.write(f'Rows: {df.shape[0]}')
st.write(f'Columns: {df.shape[1]}')

# display the dataset
if st.checkbox("Show Dataset"):
    st.write("### Enter the number of rows to view")
    rows = st.number_input("", min_value=0, value=5)
    if rows > 0:
        st.dataframe(df.head(rows))

# Text Preprocessing
# Lower casing:
df['content'] = df['content'].str.lower()
# Removing punctuations:
df['content'] = df['content'].str.lower()
# Lemmatization
df['content'] = df['content'].apply(lambda y: " ".join([Word(word).
                                                       lemmatize() for word in y.split()]))
# replace
df['content'] = df['content'].str.replace("rewardz", "reward")

# get the countPlot
st.subheader("Number of reviews for each Score")
st.success("Generating A Count Plot")
st.subheader(" Count Plot for Different Scores")
st.write(sns.countplot(df["score"]))
st.pyplot()

# Function for getting the sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
sentiment_comp = []
for row in df['content']:
    vs = analyzer.polarity_scores(row)
    sentiment_comp.append(vs)

# Creating new dataframe with sentiments
df_sentiments = pd.DataFrame(sentiment_comp)

# Merging the sentiments back to reviews dataframe
df = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
percentage_comp = []

for i in df['compound']:
    if i > 0:
        percentage = i * 100
    elif i < 0:
        percentage = -i * 100
    else:
        percentage = " "
    percentage_comp.append(percentage)
df['percentage'] = percentage_comp

# Convert scores into positive, negative and not defined sentiments using some threshold with VADER sentiment
df["Sentiment"] = df["compound"].apply(lambda compound: "positive" if compound > 0 else \
    ("negative" if compound < 0 else "not defined"))
df.drop(['neg', 'neu', 'pos', 'compound'], axis='columns', inplace=True)
st.subheader(
    'Sentiment Classification into positive, negative and neutral along with percentage of positivity/negativity')

# Select columns to display
st.subheader("Show dataset with selected columns")
# get the list of columns
columns = df.columns.tolist()
st.write("#### Select the columns to display:")
selected_cols = st.multiselect("", columns)
if len(selected_cols) > -1:
    selected_df = df[selected_cols]
    st.dataframe(selected_df)

# get the countPlot
st.subheader("How many Positive, Negative and Neutral reviews?")
st.success("Generating A Count Plot")
st.subheader(" Count Plot for Different Sentiments")
st.write(sns.countplot(df["Sentiment"]))
st.pyplot()


# Build wordcloud:
# calculating sentiments
reviews = np.array(df['content'])
size = (len(df))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
pos_reviews = ''
neg_reviews = ''
concat_reviews = ''
score_arr = []
for i in range(size):
    sentence = reviews[i]
    concat_reviews += ' %s' % sentence
    vs = analyzer.polarity_scores(sentence)
    if vs.get('compound') >= 0:
        pos_reviews += ' %s' % sentence
    else:
        neg_reviews += ' %s' % sentence
    score_arr.append(vs.get('compound'))

stopwords = set(STOPWORDS)
stopwords.add('app')


# optionally add: stopwords=STOPWORDS and change the arg below
def generate_wordcloud(text):
    wordcloud = WordCloud(relative_scaling=1.0,
                          scale=3,
                          stopwords=stopwords
                          ).generate(text)
    plt.figure(figsize=(20, 20))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot()


st.subheader("Word cloud for Negative reviews")
st.success("Generating A Negative WordCloud")
generate_wordcloud(neg_reviews)

st.subheader("Word cloud for Positive reviews")
st.success("Generating A Positive WordCloud")
generate_wordcloud(pos_reviews)

# intent analysis

# import required functions from a pretrained xlnet model
def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings - 2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids


def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=2):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

# deine a tokenizer
tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', do_lower_case=True)

# deine labels and num_label
label_cols = ['Problem in recharge', 'Problem in reward/redeem points',
              'Problem in registration/login/username/password', 'Problem with customer care service',
              'Other complaints', 'Bad/Irrelevant comments', 'Appreciation']
num_labels = len(label_cols)

# download the trained model directly, given in the repository and write the model_save_path accordingly.
model_save_path = 'classifier_model1.pt'
device = torch.device('cpu')
model = XLNetForMultiLabelSequenceClassification(num_labels)
model.load_state_dict(torch.load(model_save_path, map_location=device))

# create a function for generating the predictions along with the probabilities of each label
def generate_predictions(model, df, num_labels, device="cpu", batch_size=32):
    num_iter = math.ceil(df.shape[0] / batch_size)

    pred_probs = np.array([]).reshape(0, num_labels)

    model.to(device)
    model.eval()

    for i in range(num_iter):
        df_subset = df.iloc[i * batch_size:(i + 1) * batch_size, :]
        X = df_subset["features"].values.tolist()
        masks = df_subset["masks"].values.tolist()
        X = torch.tensor(X)
        masks = torch.tensor(masks, dtype=torch.long)
        X = X.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])

    return pred_probs

# generate the multiple intents for each review in the last 1 week along with the probabilities and then round off the probabilities to 0 & 1.
dfx = df[['content']]
dfx_text_list = dfx["content"].values
dfx_input_ids = tokenize_inputs(dfx_text_list, tokenizer, num_embeddings=250)
dfx_attention_masks = create_attn_masks(dfx_input_ids)
dfx["features"] = dfx_input_ids.tolist()
dfx["masks"] = dfx_attention_masks
pred_probs = generate_predictions(model, dfx, num_labels, device="cpu", batch_size=32)
pred_probs = np.round(pred_probs)
dfx['Problem in recharge'] = pred_probs[:, 0]
dfx['Problem in reward/redeem points'] = pred_probs[:, 1]
dfx['Problem in registration/login/username/password'] = pred_probs[:, 2]
dfx['Problem with customer care service'] = pred_probs[:, 3]
dfx['Other complaints'] = pred_probs[:, 4]
dfx['Bad/Irrelevant comments'] = pred_probs[:, 5]
dfx['Appreciation'] = pred_probs[:, 6]
dfx = dfx.drop(['features', 'masks'], axis=1)

# display results of intent analysis
st.header("Results of the Intent Analysis")
# review count for each category
st.success("Generating review count for each category..")
bar_plot = pd.DataFrame()
bar_plot['category'] = dfx.columns[1:]
bar_plot['count'] = dfx.iloc[:, 1:].sum().values
bar_plot.sort_values(['count'], inplace=True, ascending=False)
bar_plot.reset_index(inplace=True, drop=True)
st.dataframe(bar_plot)
# barplot for the same
threshold = 200
plt.figure(figsize=(15, 8))
sns.set(font_scale=1.5)
sns.set_style('whitegrid')
pal = sns.color_palette("Blues_r", len(bar_plot))
rank = bar_plot['count'].argsort().argsort()
sns.barplot(bar_plot['category'], bar_plot['count'], palette=np.array(pal[::-1])[rank])
plt.axhline(threshold, ls='--', c='red')
plt.title("Most commons categories", fontsize=24)
plt.ylabel('Number of reviews', fontsize=18)
plt.xlabel('Category', fontsize=18)
plt.xticks(rotation='vertical')
st.pyplot()

# barplot of reviews with multiple categories
st.subheader("Reviews with multiple categories")
st.success("Generating barplot..")
rowSums = dfx.iloc[:, 1:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
sns.set(font_scale=1.5)
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Number of reviews with multiple categories", fontsize=24)
plt.ylabel('Number of reviews', fontsize=18)
plt.xlabel('Number of categories', fontsize=18)
st.pyplot()

# count and dataframe of reviews which do not belong to any category
df1 = dfx[dfx['Problem in recharge'] == 0]
df1 = df1[df1['Problem in reward/redeem points'] == 0]
df1 = df1[df1['Problem in registration/login/username/password'] == 0]
df1 = df1[df1['Problem with customer care service'] == 0]
df1 = df1[df1['Other complaints'] == 0]
df1 = df1[df1['Bad/Irrelevant comments'] == 0]
df1 = df1[df1['Appreciation'] == 0]
st.header(f'Count of reviews which do not belong to any category: {len(df1)}')
df2 = df1[['content']]
if st.checkbox("Show the reviews which do not belong to any category"):
    st.write("### Enter the number of rows to view")
    rows = st.number_input("", min_value=0, value=5)
    if rows > 0:
        st.dataframe(df2.head(rows))
