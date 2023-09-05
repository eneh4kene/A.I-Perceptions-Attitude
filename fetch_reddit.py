import praw
import pandas as pd
from datetime import datetime


# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="fVDhccd53XFCO-5NkkXpZA",
    client_secret="7db8kwcr9QIs6Em83cpurnZgY-szyw",
    user_agent="A.I Perception",
    # check_for_async=False
)

# List of A.I. related subreddits
ai_subreddits = ["artificial", "MachineLearning", "datascience", "AIethics", "singularity", "chatGPT", "Openai"]

# Specify the search query
search_query = "AI OR Artificial Intelligence"

# Initialize an empty list to store the posts
posts = []

# Date range (in Unix timestamp format)
start_date = datetime(2021, 8, 1).timestamp()
end_date = datetime(2023, 8, 31).timestamp()

# Fetch posts from multiple subreddits
for subreddit_name in ai_subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.search(search_query, limit=1000):
        post_time = post.created_utc

        # Check if the post falls within the desired date range
        if start_date <= post_time <= end_date:
            posts.append({
                "title": post.title,
                "body": post.selftext,
                "upvote_ratio": post.upvote_ratio,
                "score": post.score,
                "created_utc": post.created_utc,
                "num_comments": post.num_comments,
                "subreddit": subreddit_name
            })

# Convert to a DataFrame
df_reddit = pd.DataFrame(posts)

# Save to CSV
df_reddit.to_csv("reddit_ai_posts.csv", index=False)