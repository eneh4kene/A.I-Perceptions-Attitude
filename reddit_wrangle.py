# Conducting Exploratory Data Analysis (EDA) to understand the dataset better.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Importing necessary libraries for sentiment analysis
from textblob import TextBlob
# Importing necessary libraries for thematic analysis
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
# Importing necessary libraries for topic modeling
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
# Importing necessary libraries for time series analysis
import matplotlib.dates as mdates
# Importing necessary libraries to create a dashboard-like summary of all findings and visualizations
from matplotlib.gridspec import GridSpec





# Load the dataset
dataset_path = 'reddit_ai_posts.csv'
reddit_data = pd.read_csv(dataset_path)

# Show the first few rows to understand its structure
# print(reddit_data.head())


# EDA
# Summary statistics
summary_stats = reddit_data.describe()

# Checking for missing values
missing_values = reddit_data.isnull().sum()

# Distribution of upvote ratios
plt.figure(figsize=(10, 6))
sns.histplot(reddit_data['upvote_ratio'], bins=20, kde=True)
plt.title('Distribution of Upvote Ratios')
plt.xlabel('Upvote Ratio')
plt.ylabel('Frequency')
plt.show()

# Distribution of scores
plt.figure(figsize=(10, 6))
sns.histplot(reddit_data['score'], bins=50, kde=True)
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.xlim([0, 500])  # Limiting x-axis to better visualize the distribution
# plt.show()

# Distribution of number of comments
plt.figure(figsize=(10, 6))
sns.histplot(reddit_data['num_comments'], bins=50, kde=True)
plt.title('Distribution of Number of Comments')
plt.xlabel('Number of Comments')
plt.ylabel('Frequency')
plt.xlim([0, 300])  # Limiting x-axis to better visualize the distribution
# plt.show()

# print(summary_stats, missing_values)


# Function to calculate sentiment polarity
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Calculate sentiment polarity for each Reddit post title
reddit_data['sentiment_polarity'] = reddit_data['title'].apply(calculate_sentiment)

# Visualize the distribution of sentiment polarity
plt.figure(figsize=(10, 6))
sns.histplot(reddit_data['sentiment_polarity'], bins=30, kde=True)
plt.title('Distribution of Sentiment Polarity in Reddit Post Titles')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Summary statistics for sentiment polarity
sentiment_summary = reddit_data['sentiment_polarity'].describe()
print(sentiment_summary)


# THEMATIC ANALYSIS
# Initialize CountVectorizer to convert text into a matrix of token counts
vectorizer = CountVectorizer(stop_words='english', max_features=1000)

# Fit and transform the titles into the CountVectorizer model
X = vectorizer.fit_transform(reddit_data['title'])

# Create a DataFrame for the most frequent words
word_frequency = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_sum = word_frequency.sum().sort_values(ascending=False).head(20)

# Generate a word cloud for the most frequent words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_sum)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Reddit AI Post Titles')
plt.show()

# print(word_sum)


# Analyzing upvote ratios and scores to understand the popularity or controversy of different topics

# Filter posts with high upvote ratios (greater than or equal to 0.9) and high scores (greater than or equal to the median score)
high_upvote = reddit_data['upvote_ratio'] >= 0.9
high_score = reddit_data['score'] >= reddit_data['score'].median()

# Posts that are both popular and well-received
popular_well_received_posts = reddit_data[high_upvote & high_score]

# Extracting the most frequent words in popular and well-received posts
popular_X = vectorizer.transform(popular_well_received_posts['title'])
popular_word_frequency = pd.DataFrame(popular_X.toarray(), columns=vectorizer.get_feature_names_out())
popular_word_sum = popular_word_frequency.sum().sort_values(ascending=False).head(20)

# Visualization of most frequent words in popular and well-received posts
popular_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(popular_word_sum)
plt.figure(figsize=(10, 6))
plt.imshow(popular_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Popular and Well-Received Reddit AI Post Titles')
# plt.show()

# print(popular_word_sum)


# TOPIC CATEGORIZATION

# Filtering out posts with negative sentiment scores
negative_sentiment_posts = reddit_data[reddit_data['sentiment_polarity'] < 0]['title']

# Initialize CountVectorizer for topic modeling
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)

# Fit and transform the titles into the CountVectorizer model
X_topics = vectorizer.fit_transform(negative_sentiment_posts)

# Initialize and fit LDA model
lda_model = LDA(n_components=4, random_state=42)  # We aim to find 4 topics
lda_model.fit(X_topics)

# Get the top 10 words per topic
topic_words = {}
n_top_words = 10

for topic, comp in enumerate(lda_model.components_):
    word_idx = comp.argsort()[:-n_top_words - 1:-1]
    topic_words[topic] = [vectorizer.get_feature_names_out()[i] for i in word_idx]

# print(topic_words)


# Visualizing the topics using a bar chart for each topic's top 10 keywords
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for topic, words in topic_words.items():
    ax = axes[topic]
    freq_series = pd.Series(lda_model.components_[topic], index=vectorizer.get_feature_names_out())
    ax.barh(words, freq_series.loc[words])
    ax.set_title(f'Topic {topic + 1}')
    ax.set_xlabel('Word Frequency')

plt.tight_layout()
plt.suptitle('Top 10 Keywords for Each Topic', fontsize=16, y=1.02)
# plt.show()


# Redisplaying the bar chart visualizations for each topic's top 10 keywords
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for topic, words in topic_words.items():
    ax = axes[topic]
    freq_series = pd.Series(lda_model.components_[topic], index=vectorizer.get_feature_names_out())
    ax.barh(words, freq_series.loc[words])
    ax.set_title(f'Topic {topic + 1}')
    ax.set_xlabel('Word Frequency')

plt.tight_layout()
plt.suptitle('Top 10 Keywords for Each Topic', fontsize=16, y=1.02)
plt.show()


# Filtering the dataset for posts that specifically mention terms related to jobs, employment, or opportunities
job_related_keywords = ['job', 'employment', 'opportunity', 'work', 'career', 'displace', 'replace', 'automate', 'layoff']
job_related_posts = reddit_data[reddit_data['title'].str.contains('|'.join(job_related_keywords), case=False, na=False)]

# Conducting sentiment analysis on these filtered posts
job_related_posts['sentiment_polarity'] = job_related_posts['title'].apply(calculate_sentiment)

# Visualize the distribution of sentiment polarity for job-related posts
plt.figure(figsize=(10, 6))
sns.histplot(job_related_posts['sentiment_polarity'], bins=30, kde=True)
plt.title('Distribution of Sentiment Polarity in Job-Related Reddit Post Titles')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Summary statistics for sentiment polarity in job-related posts
job_sentiment_summary = job_related_posts['sentiment_polarity'].describe()
print(job_sentiment_summary)


# Conducting thematic analysis on job-related posts to understand the nuances of public perception

# Fit and transform the titles of job-related posts into the CountVectorizer model
job_related_X = vectorizer.fit_transform(job_related_posts['title'])

# Initialize and fit LDA model for job-related posts
job_related_lda_model = LDA(n_components=4, random_state=42)  # We aim to find 4 topics
job_related_lda_model.fit(job_related_X)

# Get the top 10 words per topic for job-related posts
job_related_topic_words = {}
n_top_words = 10

for topic, comp in enumerate(job_related_lda_model.components_):
    word_idx = comp.argsort()[:-n_top_words - 1:-1]
    job_related_topic_words[topic] = [vectorizer.get_feature_names_out()[i] for i in word_idx]

# Visualizing the topics using a bar chart for each job-related topic's top 10 keywords
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for topic, words in job_related_topic_words.items():
    ax = axes[topic]
    freq_series = pd.Series(job_related_lda_model.components_[topic], index=vectorizer.get_feature_names_out())
    ax.barh(words, freq_series.loc[words])
    ax.set_title(f'Job-Related Topic {topic + 1}')
    ax.set_xlabel('Word Frequency')

plt.tight_layout()
plt.suptitle('Top 10 Keywords for Each Job-Related Topic', fontsize=16, y=1.02)
plt.show()

# print(job_related_topic_words)



# Converting the 'created_utc' column to a datetime format
reddit_data['created_datetime'] = pd.to_datetime(reddit_data['created_utc'], unit='s')

# Resampling the data to monthly frequency, taking the mean sentiment polarity for each month
time_series_data = reddit_data.resample('M', on='created_datetime').mean()['sentiment_polarity']

# Plotting the time series analysis
plt.figure(figsize=(15, 7))
plt.plot(time_series_data.index, time_series_data.values, marker='o')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()
plt.title('Evolving Attitude Towards AI Over Time')
plt.xlabel('Time (Monthly)')
plt.ylabel('Average Sentiment Polarity')
plt.grid(True)
plt.show()

# Showing a few sample data points for context
# print(time_series_data.tail())


# SUMMARY ~ DASHBOARD
# Create a figure and set up a grid
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(4, 2, figure=fig)

# Plot 1: General Sentiment Distribution
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(reddit_data['sentiment_polarity'], bins=30, kde=True, ax=ax1)
ax1.set_title('General Sentiment Distribution')
ax1.set_xlabel('Sentiment Polarity')
ax1.set_ylabel('Frequency')

# Plot 2: Job-Related Sentiment Distribution
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(job_related_posts['sentiment_polarity'], bins=30, kde=True, ax=ax2)
ax2.set_title('Job-Related Sentiment Distribution')
ax2.set_xlabel('Sentiment Polarity')
ax2.set_ylabel('Frequency')

# Plot 3: Time Series Analysis
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(time_series_data.index, time_series_data.values, marker='o')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
ax3.set_title('Evolving Attitude Towards AI Over Time')
ax3.set_xlabel('Time (Monthly)')
ax3.set_ylabel('Average Sentiment Polarity')
ax3.grid(True)

# Plot 4: Most Frequent Words in General Posts
ax4 = fig.add_subplot(gs[2, 0])
ax4.barh(list(word_sum.index), word_sum.values)
ax4.set_title('Most Frequent Words in General Posts')
ax4.set_xlabel('Frequency')
ax4.set_ylabel('Words')

# Plot 5: Most Frequent Words in Job-Related Posts
ax5 = fig.add_subplot(gs[2, 1])
job_related_freq_series = pd.Series(job_related_lda_model.components_[0], index=vectorizer.get_feature_names_out())
ax5.barh(job_related_topic_words[0], job_related_freq_series.loc[job_related_topic_words[0]])
ax5.set_title('Most Frequent Words in Job-Related Posts')
ax5.set_xlabel('Frequency')
ax5.set_ylabel('Words')

# Add titles and text to convey the story
fig.suptitle('The Evolving Public Perception Towards AI: A Reddit Analysis', fontsize=24)
fig.text(0.1, 0.48, """Key Findings:
1. General sentiment towards AI leans slightly negative.
2. Topics of concern include ethical implications, technical complexities, and research-related issues.
3. Job-related posts also indicate a slightly negative sentiment, emphasizing concerns about job displacement and career paths.
4. Public sentiment towards AI has been volatile over time, but generally leans negative.""", fontsize=16, verticalalignment="top")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

