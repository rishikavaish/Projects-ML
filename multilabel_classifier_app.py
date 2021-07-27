import os
import math

import torch
from torch.nn import BCEWithLogitsLoss
# from torch.utils.data import tensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLNetModel, XLNetLMHeadModel, XLNetConfig
from transformers import XLNetTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

import streamlit as st


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


tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', do_lower_case=True)

label_cols = ['Problem in recharge', 'Problem in reward/redeem points',
              'Problem in registration/login/username/password', 'Problem with customer care service',
              'Other complaints', 'Bad/Irrelevant comments', 'Appreciation']
num_labels = len(label_cols)

model_save_path = 'classifier_model1 (1).pt'
device = torch.device('cpu')
model = XLNetForMultiLabelSequenceClassification(num_labels)
model.load_state_dict(torch.load(model_save_path, map_location=device))


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


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# heading
html_temp = """
<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Multi-label Intent Classifier</p></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# labels
st.header("Labels to be predicted:")
c1, c2, c3, c4 = st.beta_columns([0.5, 0.5, 0.5, 0.5])
c1.button("Problem in recharge")
with c2:
    st.button("Problem in reward/redeem points")
with c3:
    st.button("Problem in customer care service")
with c4:
    st.button("Problem in registration/login")
c5, c6, c7 = st.beta_columns([0.5, 0.5, 0.5])
with c5:
    st.button("Other complaints")
with c6:
    st.button("Appreciation")
with c7:
    st.button("Bad/Irrelevant comments")

# input text
comment = st.text_input("Enter your text ")

# sentiments and intents
if st.button("Get results"):
    st.header("Classification results")
    st.success("Generating sentiment..")
    vs = analyzer.polarity_scores(comment)
    p = abs(vs['compound']) * 100
    p = np.round(p, 3)
    if vs['compound'] > 0:
        st.subheader(f'The sentiment of your text is {p}% positive!')
    elif vs['compound'] < 0:
        st.subheader(f'The sentiment of your text is {p}% negative!')
    else:
        st.subheader('The sentiment of your text is neutral')
    st.text("")
    st.success("Generating intents with there probabilities...")
    data = [[comment]]
    df1 = pd.DataFrame(data, columns=['content'])
    df1_text_list = df1["content"].values
    df1_input_ids = tokenize_inputs(df1_text_list, tokenizer, num_embeddings=250)
    df1_attention_masks = create_attn_masks(df1_input_ids)
    df1["features"] = df1_input_ids.tolist()
    df1["masks"] = df1_attention_masks
    pred_probs = generate_predictions(model, df1, num_labels, device="cpu", batch_size=1)
    pred_probs = pred_probs * 100
    pred_probs = np.round(pred_probs, 2)
    probsList = [item for elem in pred_probs for item in elem]
    THRESHOLD = 50
    st.header("Your text is classified as")
    if probsList[0] < THRESHOLD and probsList[1] < THRESHOLD and probsList[2] < THRESHOLD and probsList[3] < THRESHOLD and probsList[4] < THRESHOLD and probsList[5] < THRESHOLD and probsList[6] < THRESHOLD:
        st.subheader("Your text does not belong to any category!")
    else:
        for label, prediction in zip(label_cols, probsList):
            if prediction < THRESHOLD:
                continue
            st.subheader(f"{prediction}%  {label}")

st.sidebar.header("About App")
st.sidebar.info(
    "A Multi-label intent classification Project which will predict the sentiment and multiple intents of the given text and also predicts the probability of each label.")
st.sidebar.text("Built with Streamlit")
st.sidebar.header("For Any Queries/Suggestions, Please reach out at :")
st.sidebar.info("rishikavaish321@gmail.com")
