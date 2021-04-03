# Import required libraries
import pandas as pd
import hvplot.pandas
import PyPDF2
import textract
import re
import string
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
%matplotlib inline

# Open pdf file
pdfFileObj = open('cover-letter.pdf','rb')

# Read file
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# Get total number of pages
num_pages = pdfReader.numPages

# Initialize a count for the number of pages
# (Ideally, cover letters should be one page long)
count = 0

# Initialize a text empty string variable
text = ""

# Extract text from every page on the file
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()
    
# Create a list of the words
words = word_tokenize(text)

# Convert the words to lowercase
words = [word.lower() for word in words]

# Remove the punctuation
words = [word for word in words if word not in punctuation]

# Create a list of stopwords
stop = stopwords.words('english')

# Remove the stopwords
words = [word for word in words if word not in stop]

# Join tokens back into a single string
text = (" ").join(words)

# Download VADER Lexicon
nltk.download('vader_lexicon')

# Instantiate the VADER sentiment analyzer in a new variable
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiment_scores = analyzer.polarity_scores(text)

# Create a data frame with the sentiment scores
df = pd.DataFrame(sentiment_scores, index=[pdfFileObj.name])

# Rename data frame columns
columns = ['Negative','Neutral','Positive','Compound']
df.columns = columns

# Create hvplot bar chart to visualize the sentiments results
df.hvplot.bar(title='Cover Letter Sentiment Analysis Results', xlabel='Sentiment', ylabel='Score')
