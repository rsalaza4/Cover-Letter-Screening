# Import required libraries
import pandas as pd
import hvplot.pandas
import PyPDF2
import textract
import re
import string
import nltk
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
    
# Convert all strings to lowercase
text = text.lower()

# Remove numbers
text = re.sub(r'\d+','',text)

# Remove punctuation
text = text.translate(str.maketrans('','',string.punctuation))

# Remove row jumps
text = text.replace('\n','')

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
