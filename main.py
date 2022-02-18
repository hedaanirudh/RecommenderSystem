import pandas as pd
from dataloader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class Recommend:

    def __init__(self):
        self.data_loader = DataLoader()

    def get_data(self, user_input=None):
        # Get Data from ratings.csv
        mat = self.data_loader.load_data(user_input)
        return mat

    # Function to calculate row wise cosine similarity for a sparse matrix
    def get_row_similarity(self, matrix):
        matrix_sparse = sparse.csr_matrix(matrix)
        similarity_matrix = cosine_similarity(matrix_sparse)
        return similarity_matrix

    # Function to calculate similarity between users (i.e., rows of matrix)
    def get_user_similarities(self, mat):
        user_similarities = self.get_row_similarity(mat)
        user_similarities = pd.DataFrame(user_similarities, columns=mat.index).set_index(mat.index)
        return user_similarities

    # Function to calculate similarity between movies (i.e., columns of matrix)
    def get_movie_similarities(self, mat):
        mat_transpose = mat.transpose()
        movie_similarities = self.get_row_similarity(mat_transpose)
        movie_similarities = pd.DataFrame(movie_similarities, columns=mat.columns).set_index(mat.columns)
        return movie_similarities

    # Function to create the user and movie similarity matrix from the original csv
    def get_similarity_matrix_org(self):
        data = self.get_data()
        mat = self.data_loader.transform_data(data)
        user_similarities = self.get_user_similarities(mat)
        movie_similarities = self.get_movie_similarities(mat)
        return user_similarities, movie_similarities

    # Function to recalculate user similarity matrix and return recommendations
    def get_recommendations_for_users(self, user_input):
        data = self.get_data(user_input)
        mat = self.data_loader.transform_data(data)
        user_similarities = self.get_user_similarities(mat)
        user_id = user_input[0]["userId"]

        # Getting the 2 most similar users to the new user
        similar_users = user_similarities[user_id].sort_values(ascending=False)[1:3]
        similar_user_ids = [index for index, value in similar_users.items()]

        # Getting the movies they rated
        relevant_user_movies = data.loc[data['userId'].isin(similar_user_ids)]
        relevant_movie_ids = set(relevant_user_movies['movieId'])
        my_movie_ids = set(data.loc[data['userId'] == user_id]['movieId'])

        # Listing top 5 movies that they have rated and haven't been watched by the new user
        recommendations = \
        relevant_user_movies.loc[relevant_user_movies['movieId'].isin(relevant_movie_ids - my_movie_ids)].sort_values(
            by=['rating'], ascending=False)['title'].unique()[:5]
        return recommendations


if __name__ == '__main__':
    rec = Recommend()
    # Get the existing user and movie similarity matrices
    user_similarities, movie_similarities = rec.get_similarity_matrix_org()

    user_input = [{"userId": 200, "movieId": 1, "rating": 5},
                  {"userId": 200, "movieId": 4246, "rating": 2},
                  {"userId": 200, "movieId": 260, "rating": 4},
                  {"userId": 200, "movieId": 231, "rating": 2.5},
                  {"userId": 200, "movieId": 356, "rating": 4},
                  {"userId": 200, "movieId": 480, "rating": 5},
                  {"userId": 200, "movieId": 593, "rating": 4.5},
                  {"userId": 200, "movieId": 318, "rating": 4},
                  {"userId": 200, "movieId": 589, "rating": 4},
                  {"userId": 200, "movieId": 153, "rating": 2},
                  {"userId": 200, "movieId": 1196, "rating": 4},
                  {"userId": 200, "movieId": 1270, "rating": 4}]
    print("My rating for a few movies - ")
    print(user_input)

    # Recalculate user similarity based on new inputs and return recommendations based on same
    recommendations = rec.get_recommendations_for_users(user_input)
    print("Recommendations for me - " + str(recommendations))
