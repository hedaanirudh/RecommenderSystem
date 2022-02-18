import pandas as pd
import itertools
import numpy as np


class DataLoader:
    def __init__(self):
        self.file_name = "ratings.csv"
        self.file_df = pd.read_csv(self.file_name)
        self.movie_df = pd.read_csv('movies.csv')[['movieId', 'title']].set_index('movieId')

    def load_data(self, user_input=None):
        # Loads data
        group = self.file_df.groupby('userId')
        df = pd.concat([x for _, x in itertools.islice(group, 0, 15)])
        if user_input:
            df = df.append(user_input, ignore_index=True, sort=False)
        movie_count = df.groupby('movieId')['rating'].count()
        top_movies = movie_count.sort_values(ascending=False)[:50]
        filtered_file = df.join(top_movies, on='movieId', how='inner', rsuffix='_r')
        filtered_file = filtered_file.join(self.movie_df, on='movieId', how='inner', rsuffix='_r')[['userId', 'movieId', 'rating', 'title']]
        return filtered_file

    def transform_data(self, df):
        # Create user vs movie matrix
        transformed_df = pd.crosstab(df.userId, df.title, df.rating, aggfunc=np.sum).fillna(0)
        return transformed_df
