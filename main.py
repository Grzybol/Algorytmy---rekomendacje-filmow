import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def map_users_to_indices(users, specific_user=None):
    mapping = {uid: idx for idx, uid in enumerate(users)}
    return mapping if specific_user is None else mapping[specific_user]

def map_movies_to_indices(movies, specific_movie=None):
    mapping = {mid: idx for idx, mid in enumerate(movies)}
    return mapping if specific_movie is None else mapping[specific_movie]

def load_data():
    data = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])
    unique_users = sorted(data['userId'].unique())
    unique_movies = sorted(data['movieId'].unique())
    
    ratings_matrix = np.zeros((len(unique_users), len(unique_movies)))

    user_index = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_index = {mid: idx for idx, mid in enumerate(unique_movies)}

    indices_users = data['userId'].map(user_index)
    indices_movies = data['movieId'].map(movie_index)
    ratings_matrix[indices_users, indices_movies] = data['rating']

    return ratings_matrix, unique_users, unique_movies

def get_neighbors(user, matrix, k, users):
    user_idx = map_users_to_indices(users, user)

    model = NearestNeighbors(n_neighbors=k + 1, metric='correlation')
    model.fit(matrix)

    distances, indices = model.kneighbors(matrix[user_idx].reshape(1, -1), return_distance=True)

    neighbors = [users[i] for i in indices.flatten() if users[i] != user]

    return neighbors[:k]

def suggest_movie(user, matrix, neighbors, similarities, users, movies):
    user_idx = map_users_to_indices(users, user)
    unrated_movies_idx = np.where(matrix[user_idx] == 0)[0]

    movie_recommendations = {}

    for idx in unrated_movies_idx:
        movie = movies[idx]
        ratings = []
        similarity_scores = []

        for neighbor in neighbors:
            neighbor_idx = map_users_to_indices(users, neighbor)
            neighbor_rating = matrix[neighbor_idx, idx]

            if neighbor_rating > 0:
                rating_similarity = similarities[user_idx, neighbor_idx]
                similarity_scores.append(rating_similarity)
                ratings.append(neighbor_rating)

        if ratings:
            weighted_ratings = np.dot(ratings, similarity_scores)
            total_similarity = np.sum(similarity_scores)
            movie_recommendations[movie] = weighted_ratings / total_similarity

    return max(movie_recommendations, key=movie_recommendations.get)

def main():
    user = 1
    k = 10

    rating_matrix, users, movies = load_data()
    similarity_matrix = 1 - pairwise_distances(rating_matrix, metric='correlation')
    np.fill_diagonal(similarity_matrix, 0)

    closest_neighbors = get_neighbors(user, rating_matrix, k, users)
    print(f"Closest neighbors for user {user}: {closest_neighbors}")
    recommended_movie = suggest_movie(user, rating_matrix, closest_neighbors, similarity_matrix, users, movies)
    print("Recommended movie:", recommended_movie)

if __name__ == '__main__':
    main()
