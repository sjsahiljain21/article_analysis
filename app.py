#!/usr/bin/env python
# coding: utf-8

# In[19]:


import flair
from newsfetch.news import newspaper
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# In[22]:


# from jupyter_dash import JupyterDash
# nltk.download('vader_lexicon')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sid = SentimentIntensityAnalyzer()

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
# from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State


# In[23]:


flair_sentiment = flair.models.TextClassifier.load('en-sentiment') 

def flair_polarity_score(text):
    if len(text) < 5:
        return 0
    text_sentiment = flair.data.Sentence(text)
    flair_sentiment.predict(text_sentiment)
    value = text_sentiment.labels[0].to_dict()['value'] 
    if value == 'POSITIVE':
        result = text_sentiment.to_dict()['labels'][0]['confidence']
    else:
        result = -(text_sentiment.to_dict()['labels'][0]['confidence'])
    return result


# In[24]:


def get_scores_flair(sentences):
    """ Call predict on every sentence of a text """
    results = []
    
    for i in range(0, len(sentences)): 
        results.append(flair_polarity_score(sentences[i]))
    return results


# In[25]:


def get_mean(scores):
    result = sum(scores)/len(scores)
    return result


# In[ ]:





# In[8]:





# In[ ]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("How awesome is this?"),
    html.Div(dcc.Input(id = 'url-link', placeholder = 'Enter URL', size = '80', type = 'url', value = 'https://www.hindustantimes.com/hollywood/when-chadwick-boseman-was-fired-from-tv-show-for-questioning-producers-replaced-by-future-black-panther-co-star-michael-b-jordan/story-W0lt3fNQ2y45C5SDDhuZNJ.html')),
    html.Button('Submit', id='submit', n_clicks=0),
    html.Div(html.H6("Sentiment Score")),
    html.Div(id='article-sentiment'),
    html.Div(html.H6("Publish Date")),
    html.Div(id='article-publish_date'),
    html.Div(html.H6("Headline")),
    html.Div(id = 'article-headline'),
    html.Div(html.H6("Body")),
    html.Div(id = 'article-body'),
    html.Div(html.H6("Summary")),
    html.Div(id = 'article-summary'),
    html.Div(html.H6("Keywords")),
    html.Div(id = 'article-keywords'),
    html.Div(html.H6("Authors")),
    html.Div(id = 'article-authors')
])

@app.callback([Output('article-sentiment', 'children'),
               Output('article-publish_date', 'children'),
              Output('article-headline', 'children'),
              Output('article-body', 'children'),
              Output('article-summary', 'children'),
              Output('article-keywords', 'children'),
              Output('article-authors', 'children')],
              [Input('submit', 'n_clicks')],
             [State('url-link', 'value')])

def update_sentiment(n_clicks, input_value):
    news = newspaper(input_value)
    publish_date = news.date_publish
    headline = news.headline
    body = news.article
    sentiment = get_mean(get_scores_flair(sent_tokenize(body)))
    summary = news.summary
    keywords = ', '.join(news.keywords)
    authors = ', '.join(news.authors)
    return sentiment, publish_date, headline, body, summary, keywords, authors

if __name__ == '__main__':
    app.run_server(debug=True)

