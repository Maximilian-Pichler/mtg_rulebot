SELECT body, subreddit
FROM `fh-bigquery.reddit_comments.2019_*`
where subreddit like "%mtg%"
and body like "%[[%]]%"