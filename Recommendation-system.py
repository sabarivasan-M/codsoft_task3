import pandas as pd
import numpy as np

# Sample dataset: user ratings for movies
ratings_dict = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movie_id': [101, 102, 103, 101, 104, 105, 102, 104, 106, 103, 105, 106],
    'rating': [5, 4, 3, 5, 4, 2, 4, 3, 5, 2, 5, 4]
}

ratings_df = pd.DataFrame(ratings_dict)
print(ratings_df)
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
print(user_item_matrix)
from sklearn.metrics.pairwise import cosine_similarity

# Fill NaN with 0s to compute similarities
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print(user_similarity_df)
def recommend_movies(user_id, user_item_matrix, user_similarity_df, num_recommendations=3):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Aggregate ratings of similar users
    weighted_ratings = np.dot(similar_users, user_item_matrix_filled)

    # Create a DataFrame of recommendations
    recommendations = pd.DataFrame(weighted_ratings, index=user_item_matrix_filled.columns, columns=['score'])

    # Remove movies already rated by the user
    recommendations = recommendations[~recommendations.index.isin(user_ratings.dropna().index)]

    # Return the top N recommendations
    return recommendations.sort_values('score', ascending=False).head(num_recommendations)

# Recommend movies for a specific user (e.g., user 1)
recommended_movies = recommend_movies(1, user_item_matrix, user_similarity_df)
print(recommended_movies)