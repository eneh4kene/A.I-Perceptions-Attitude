# Importing necessary libraries to examine the dataset
import pandas as pd
import numpy as np
# Importing necessary libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Importing necessary libraries for sentiment analysis
from textblob import TextBlob
# Importing necessary libraries for text mining and word cloud generation
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import collections
# Importing necessary libraries for generating dashboards
import matplotlib.gridspec as gridspec


# Load the dataset
dataset_path = 'dataset2.csv'
df = pd.read_csv(dataset_path)

# Display basic information about the dataset
dataset_info = df.info()
dataset_head = df.head()
print(dataset_info, dataset_head)

# DATA CLEANING
# Importing necessary libraries for data cleaning and preprocessing

# Convert columns to appropriate data types
df['is_answered'] = df['is_answered'].astype(bool)
df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce')
df['answer_count'] = pd.to_numeric(df['answer_count'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['creation_date'] = pd.to_datetime(df['creation_date'], unit='s', errors='coerce')

# Filter rows that have AI-related tags
ai_related_tags = ['ai', 'artificial-intelligence', 'machine-learning', 'deep-learning', 'nlp', 'neural-network']
df['is_ai_related'] = df['tags'].apply(lambda x: any(tag in x for tag in ai_related_tags))

# Drop rows that are not AI-related
df_ai_related = df[df['is_ai_related']].copy()

# Drop NaN values for relevant columns
df_ai_related.dropna(subset=['view_count', 'answer_count', 'score', 'creation_date'], inplace=True)

# Display cleaned data summary
cleaned_data_info = df_ai_related.info()
cleaned_data_head = df_ai_related.head()

print(cleaned_data_info, cleaned_data_head)


# EDA
# Setting style
sns.set_style("whitegrid")

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Plot 1: Distribution of view_count
sns.histplot(df_ai_related['view_count'], bins=50, kde=False, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of View Counts')
axes[0, 0].set_xlabel('View Count')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Distribution of answer_count
sns.histplot(df_ai_related['answer_count'], bins=50, kde=False, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Answer Counts')
axes[0, 1].set_xlabel('Answer Count')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Distribution of Scores
sns.histplot(df_ai_related['score'], bins=50, kde=False, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Scores')
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Questions over Time
df_ai_related['creation_date'].dt.to_period("M").value_counts().sort_index().plot(ax=axes[1, 1], kind='line')
axes[1, 1].set_title('Number of AI-related Questions Over Time')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Number of Questions')

plt.tight_layout()
# plt.show()

# SENTIMENT ANALYSIS
# Function to calculate sentiment polarity
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Calculate sentiment polarity for each question title
df_ai_related['sentiment_polarity'] = df_ai_related['title'].apply(calculate_sentiment)

# Create a new column to classify the sentiment as Positive, Neutral, or Negative
df_ai_related['sentiment'] = np.where(df_ai_related['sentiment_polarity'] > 0, 'Positive',
                                     np.where(df_ai_related['sentiment_polarity'] < 0, 'Negative', 'Neutral'))

# Plot the sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df_ai_related, order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Distribution in AI-related Questions')
plt.xlabel('Sentiment')
plt.ylabel('Number of Questions')
# plt.show()

# Show the first few rows with sentiment analysis
df_ai_related[['title', 'sentiment_polarity', 'sentiment']].head()

# TEXT-MINING
# Step 1: Prepare the text data for analysis
# Combine all the titles into a single string
text_data = ' '.join(df_ai_related['title'])

# Step 2: Generate a word cloud to visualize the most frequently occurring words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of AI-related Questions')
# plt.show()

# Step 3: Perform text mining to identify common themes or topics
# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=20)

# Fit and transform the title text
X = vectorizer.fit_transform(df_ai_related['title'])

# Sum up the counts of each vocabulary word
word_freq = np.sum(X.toarray(), axis=0)

# Get feature names
feature_names = np.array(vectorizer.get_feature_names_out())

# Sort by frequency
sorted_indices = np.argsort(-word_freq)

# Create a dictionary of words and their corresponding frequencies
word_freq_dict = collections.OrderedDict((feature_names[i], word_freq[i]) for i in sorted_indices)

# print(word_freq_dict)

# TOPICAL CATEGORIZATION
# Step 1: Create a list of keywords for each category ('Ethical', 'Complexity', 'Technical')
ethical_keywords = ['ethic', 'moral', 'bias', 'fairness', 'privacy', 'trust', 'responsible', 'accountable']
complexity_keywords = ['complex', 'difficult', 'challenge', 'problem', 'issue', 'complicated']
technical_keywords = ['model', 'tensorflow', 'keras', 'pytorch', 'algorithm', 'code', 'debug', 'error', 'data']

# Step 2: Search for these keywords in the question titles and categorize each title
def categorize_title(title):
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in ethical_keywords):
        return 'Ethical'
    elif any(keyword in title_lower for keyword in complexity_keywords):
        return 'Complexity'
    elif any(keyword in title_lower for keyword in technical_keywords):
        return 'Technical'
    else:
        return 'Other'

# Apply the function to categorize titles
df_ai_related['category'] = df_ai_related['title'].apply(categorize_title)

# Step 3: Analyze the sentiment distribution within each category
category_sentiment_distribution = df_ai_related.groupby(['category', 'sentiment']).size().reset_index(name='count')

# Plotting the sentiment distribution within each category
plt.figure(figsize=(12, 8))
sns.barplot(x='category', y='count', hue='sentiment', data=category_sentiment_distribution)
plt.title('Sentiment Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Number of Questions')
# plt.show()

# print(category_sentiment_distribution)

# DASHBOARD
# Create a dashboard using Matplotlib's gridspec
fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[0.4, 0.4, 0.2])
fig.suptitle('Dashboard: Attitudes and Sentiments Toward AI Technologies', fontsize=20)

# Plot 1: Distribution of View Counts
ax0 = plt.subplot(gs[0])
sns.histplot(df_ai_related['view_count'], bins=50, kde=False, ax=ax0)
ax0.set_title('Distribution of View Counts')
ax0.set_xlabel('View Count')
ax0.set_ylabel('Frequency')

# Plot 2: Distribution of Answer Counts
ax1 = plt.subplot(gs[1])
sns.histplot(df_ai_related['answer_count'], bins=50, kde=False, ax=ax1)
ax1.set_title('Distribution of Answer Counts')
ax1.set_xlabel('Answer Count')
ax1.set_ylabel('Frequency')

# Plot 3: Sentiment Distribution by Category
ax2 = plt.subplot(gs[2])
sns.barplot(x='category', y='count', hue='sentiment', data=category_sentiment_distribution, ax=ax2)
ax2.set_title('Sentiment Distribution by Category')
ax2.set_xlabel('Category')
ax2.set_ylabel('Number of Questions')

# Plot 4: Word Cloud of AI-related Questions
ax3 = plt.subplot(gs[3])
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis('off')
ax3.set_title('Word Cloud of AI-related Questions')

# Plot 5: Distribution of Scores
ax4 = plt.subplot(gs[4])
sns.histplot(df_ai_related['score'], bins=50, kde=False, ax=ax4)
ax4.set_title('Distribution of Scores')
ax4.set_xlabel('Score')
ax4.set_ylabel('Frequency')

# Add narrative for the dashboard
narrative = '''
1. Distribution of View Counts: Most questions have less than 2,000 views, indicating that AI topics may be specialized.
2. Distribution of Answer Counts: Most questions have fewer than 5 answers, suggesting a need for more expert engagement.
3. Sentiment Distribution by Category: Varied sentiments across different categories ('Ethical', 'Complexity', 'Technical') indicate diverse attitudes.
4. Word Cloud: Frequent terms like "model," "error," and "data" signify key areas of focus in AI discussions.
5. Distribution of Scores: The balanced score distribution suggests mixed sentiments toward AI-related questions.
'''
ax5 = plt.subplot(gs[5])
ax5.axis('off')
ax5.text(0.5, 0.5, narrative, wrap=True, horizontalalignment='center', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


