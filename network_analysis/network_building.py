import praw

reddit = praw.Reddit(client_id='LXzg4PrQwqaD1g',
                     client_secret=,
                     user_agent='redhairedcelt')
print(reddit.read_only)
#%%
for submission in reddit.subreddit('learnpython').hot(limit=10):
    print(submission.title)


