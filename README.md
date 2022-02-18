# RecommenderSystem

This is a simple Movie Recommender project that employs Cosine similarity based collaborative filtering.
Cosine similarity helps calculate similarity between users based on their ratings for different movies and similarly helps identify similarly rated movies based on their ratings by different users.

When a new user rates a few movies according to his taste, the user similarity matrix is calculated and users similar to the new user are identified. The top 5 moovies rated by the similar users would be returned as the recommendation for the new user.


Project Set-Up (Pre-requisites: Python3)

1. Clone the project in your local machine.
2. Open terminal and go to the directory where you cloned the project.
3. Install dependent packages - 'pip install -r requirements.txt'
4. Run Project - 'python main.py'

The default setting uses only 15 users and top 50 movies to create similarity matrix. You can change this as per your need in dataloader.py
The sample dataset has been downloaed from - http://files.grouplens.org/datasets/movielens/ml-100k.zip
(It is included in the repository)

References - 
https://realpython.com/build-recommendation-engine-collaborative-filtering/
https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
