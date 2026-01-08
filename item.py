import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gc
import os
from collections import defaultdict


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering
    Optimized cho 33M ratings, 87K movies

    Ý tưởng:
    - Phim giống nhau nếu được rated bởi users giống nhau
    - Không cần biết nội dung phim (genres/tags)
    """

    def __init__(self, ratings_path, min_common_users=3, top_k_similar=50):
        """
        Args:
            ratings_path: Path to ratings.csv
            min_common_users: Tối thiểu bao nhiêu users chung để tính similarity
            top_k_similar: Lưu top K phim tương tự cho mỗi phim
        """
        print("Loading ratings...")

        # THAY ĐỔI 1: Kiểm tra file tồn tại
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

        self.ratings = pd.read_csv(ratings_path)

        self.min_common_users = min_common_users
        self.top_k_similar = top_k_similar

        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

        # Item similarity (chỉ lưu top K)
        self.item_similarity = {}  # {movieId: [(similar_movieId, score), ...]}

        # User-Item matrix (sparse)
        self.user_item_matrix = None

        # Statistics
        self.movie_stats = {}  # {movieId: {'mean_rating', 'count'}}

    def prepare_data(self):
        """
        Chuẩn bị dữ liệu và tạo sparse matrix
        """
        print("\n" + "=" * 70)
        print("PREPARING DATA")
        print("=" * 70)

        # THAY ĐỔI 2: Thêm validation
        required_cols = ['userId', 'movieId', 'rating']
        missing_cols = set(required_cols) - set(self.ratings.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Lọc users và movies có ít ratings
        print("Filtering sparse users/movies...")
        user_counts = self.ratings['userId'].value_counts()
        movie_counts = self.ratings['movieId'].value_counts()

        # Giữ users có ít nhất 5 ratings
        valid_users = user_counts[user_counts >= 5].index
        # Giữ movies có ít nhất 3 ratings
        valid_movies = movie_counts[movie_counts >= 3].index

        self.ratings = self.ratings[
            self.ratings['userId'].isin(valid_users) &
            self.ratings['movieId'].isin(valid_movies)
            ]

        print(f"After filtering:")
        print(f"  Users: {self.ratings['userId'].nunique():,}")
        print(f"  Movies: {self.ratings['movieId'].nunique():,}")
        print(f"  Ratings: {len(self.ratings):,}")

        # Create mappings
        print("\nCreating index mappings...")
        unique_users = self.ratings['userId'].unique()
        unique_movies = self.ratings['movieId'].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}

        self.movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}

        # Map to indices
        self.ratings['user_idx'] = self.ratings['userId'].map(self.user_to_idx)
        self.ratings['movie_idx'] = self.ratings['movieId'].map(self.movie_to_idx)

        # Movie statistics (cho prediction)
        print("\nComputing movie statistics...")
        movie_stats_df = (
            self.ratings.groupby('movieId')['rating']
            .agg(['mean', 'count'])
            .reset_index()
        )

        # THAY ĐỔI 3: Convert to dict format dễ access hơn
        self.movie_stats = {}
        for _, row in movie_stats_df.iterrows():
            self.movie_stats[row['movieId']] = {
                'mean': row['mean'],
                'count': row['count']
            }

        # Create sparse matrix
        print("\nCreating sparse User-Item matrix...")
        n_users = len(self.user_to_idx)
        n_movies = len(self.movie_to_idx)

        self.user_item_matrix = csr_matrix(
            (
                self.ratings['rating'].values,
                (self.ratings['user_idx'].values, self.ratings['movie_idx'].values)
            ),
            shape=(n_users, n_movies)
        )

        print(f"Matrix shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {100 * (1 - self.user_item_matrix.nnz / (n_users * n_movies)):.2f}%")
        print(f"Memory: {self.user_item_matrix.data.nbytes / 1024 / 1024:.1f} MB")

        gc.collect()

    def compute_item_similarity(self, batch_size=500):
        """
        Tính item-item similarity (OPTIMIZED)
        Chỉ lưu top K similar items
        """
        print("\n" + "=" * 70)
        print("COMPUTING ITEM-ITEM SIMILARITY")
        print("=" * 70)

        # THAY ĐỔI 4: Validate matrix exists
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not prepared. Call prepare_data() first.")

        n_movies = self.user_item_matrix.shape[1]

        # Transpose: item-user matrix
        item_user_matrix = self.user_item_matrix.T.tocsr()

        print(f"Processing {n_movies:,} movies in batches of {batch_size}...")

        self.item_similarity = {}

        # Process in batches
        for start_idx in range(0, n_movies, batch_size):
            end_idx = min(start_idx + batch_size, n_movies)

            if start_idx % 5000 == 0:
                print(f"  Progress: {start_idx:,}/{n_movies:,} ({100 * start_idx / n_movies:.1f}%)")

            # Tính similarity cho batch này vs ALL items
            batch = item_user_matrix[start_idx:end_idx]
            similarities = cosine_similarity(batch, item_user_matrix, dense_output=False)

            # Lưu top K cho mỗi item trong batch
            for i in range(similarities.shape[0]):
                movie_idx = start_idx + i
                movie_id = self.idx_to_movie[movie_idx]

                # Lấy similarity scores
                sim_scores = similarities[i].toarray().flatten()

                # Loại bỏ chính nó
                sim_scores[movie_idx] = 0

                # Lấy top K
                top_indices = np.argpartition(sim_scores, -self.top_k_similar)[-self.top_k_similar:]
                top_indices = top_indices[np.argsort(-sim_scores[top_indices])]

                # Lọc similarity > 0
                valid_indices = top_indices[sim_scores[top_indices] > 0]

                # Lưu
                similar_items = [
                    (self.idx_to_movie[idx], float(sim_scores[idx]))
                    for idx in valid_indices
                ]

                self.item_similarity[movie_id] = similar_items

            gc.collect()

        print(f"\nCompleted! Stored top-{self.top_k_similar} similar items for each movie.")

        # Statistics
        avg_similar = np.mean([len(v) for v in self.item_similarity.values()])
        print(f"Average similar items per movie: {avg_similar:.1f}")

    def get_similar_items(self, movie_id, top_n=10):
        """
        Lấy top N phim tương tự
        """
        if movie_id not in self.item_similarity:
            return None

        similar = self.item_similarity[movie_id][:top_n]
        return similar

    def predict_rating(self, user_id, movie_id):
        """
        Dự đoán rating của user cho movie
        Dựa trên ratings của user cho các phim tương tự
        """
        # THAY ĐỔI 5: Better error handling
        if user_id not in self.user_to_idx:
            # User mới: trả về average rating của movie
            if movie_id in self.movie_stats:
                return self.movie_stats[movie_id]['mean']
            return 3.0  # Default

        if movie_id not in self.movie_to_idx:
            # Movie mới: không dự đoán được
            return None

        # Lấy phim tương tự
        similar_items = self.item_similarity.get(movie_id, [])

        if not similar_items:
            # Không có similar items: trả về average
            if movie_id in self.movie_stats:
                return self.movie_stats[movie_id]['mean']
            return 3.0

        # Lấy ratings của user cho similar items
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        user_rated_movies = set(user_ratings['movieId'].values)

        numerator = 0
        denominator = 0

        for similar_movie_id, similarity in similar_items:
            if similar_movie_id in user_rated_movies:
                rating = user_ratings[
                    user_ratings['movieId'] == similar_movie_id
                    ]['rating'].values[0]

                numerator += similarity * rating
                denominator += abs(similarity)

        if denominator == 0:
            # User chưa rate phim nào tương tự
            if movie_id in self.movie_stats:
                return self.movie_stats[movie_id]['mean']
            return 3.0

        predicted = numerator / denominator
        return max(0.5, min(5.0, predicted))  # Clamp [0.5, 5.0]

    def recommend_for_user(self, user_id, top_n=10, exclude_watched=True):
        """
        Gợi ý phim cho user (Item-Based CF)

        THAY ĐỔI 6: Return DataFrame thay vì string/print
        """
        if user_id not in self.user_to_idx:
            return pd.DataFrame()  # Return empty DataFrame instead of string

        # Lấy phim user đã xem
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        watched_movies = set(user_ratings['movieId'].values)

        # Tìm candidate movies từ similar items
        candidates = defaultdict(float)

        for _, row in user_ratings.iterrows():
            movie_id = row['movieId']
            rating = row['rating']

            # Lấy phim tương tự
            similar_items = self.item_similarity.get(movie_id, [])

            for similar_movie_id, similarity in similar_items:
                if exclude_watched and similar_movie_id in watched_movies:
                    continue

                # Weighted by user's rating và similarity
                candidates[similar_movie_id] += similarity * rating

        if not candidates:
            return pd.DataFrame()  # Return empty DataFrame

        # Sort
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n * 3]  # Lấy nhiều hơn để predict

        # Predict ratings
        predictions = []
        for movie_id, score in sorted_candidates:
            pred_rating = self.predict_rating(user_id, movie_id)
            if pred_rating is not None:
                predictions.append({
                    'movieId': movie_id,
                    'predicted_rating': pred_rating,
                    'score': score
                })

        # Sort by predicted rating
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        predictions = predictions[:top_n]

        # Format results
        results = pd.DataFrame(predictions)
        if len(results) > 0:
            results['rank'] = range(1, len(results) + 1)

        return results

    def save_model(self, filepath='item_cf_model.pkl'):
        """
        Lưu model (chỉ lưu item_similarity, không lưu matrix)
        """
        print(f"\nSaving model to {filepath}...")

        # THAY ĐỔI 7: Tạo directory nếu chưa tồn tại
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        model_data = {
            'item_similarity': self.item_similarity,
            'movie_stats': self.movie_stats,
            'movie_to_idx': self.movie_to_idx,
            'idx_to_movie': self.idx_to_movie,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'top_k_similar': self.top_k_similar,
            'min_common_users': self.min_common_users  # THAY ĐỔI 8: Lưu config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        file_size = os.path.getsize(filepath) / 1024 / 1024
        print(f"Model saved! Size: {file_size:.1f} MB")

    def load_model(self, filepath='item_cf_model.pkl'):
        """
        Load model
        """
        # THAY ĐỔI 9: Better error handling
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        print(f"\nLoading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.item_similarity = model_data['item_similarity']
        self.movie_stats = model_data['movie_stats']
        self.movie_to_idx = model_data['movie_to_idx']
        self.idx_to_movie = model_data['idx_to_movie']
        self.user_to_idx = model_data['user_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.top_k_similar = model_data['top_k_similar']
        self.min_common_users = model_data.get('min_common_users', 3)  # Backward compatible

        print("Model loaded successfully!")
        print(f"  Movies: {len(self.movie_to_idx):,}")
        print(f"  Users: {len(self.user_to_idx):,}")
        print(f"  Similarity items: {len(self.item_similarity):,}")


# ===== USAGE =====
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("ITEM-BASED COLLABORATIVE FILTERING")
    print("Optimized for 33M+ ratings, 87K movies")
    print("=" * 70)

    # Initialize
    start = time.time()
    recommender = ItemBasedCF(
        ratings_path="ml-latest/ratings.csv",
        min_common_users=3,
        top_k_similar=50  # Lưu 50 phim tương tự cho mỗi phim
    )

    # Prepare data
    recommender.prepare_data()

    # Compute similarity (tốn thời gian nhất)
    recommender.compute_item_similarity(batch_size=500)

    build_time = time.time() - start
    print(f"\nTotal build time: {build_time / 60:.1f} minutes")

    # Save model
    recommender.save_model('models/item_cf_model.pkl')

    # Test 1: Similar items
    print("\n" + "=" * 70)
    print("TEST 1: SIMILAR ITEMS")
    print("=" * 70)

    test_movie_id = 1  # Toy Story
    similar = recommender.get_similar_items(test_movie_id, top_n=10)

    if similar:
        print(f"\nTop 10 movies similar to Movie {test_movie_id}:")
        for i, (movie_id, score) in enumerate(similar, 1):
            print(f"{i:2d}. Movie {movie_id:6d} - Similarity: {score:.4f}")

    # Test 2: Predict rating
    print("\n" + "=" * 70)
    print("TEST 2: RATING PREDICTION")
    print("=" * 70)

    test_user_id = 1
    test_movie_id = 100
    predicted = recommender.predict_rating(test_user_id, test_movie_id)
    print(f"\nPredicted rating for User {test_user_id}, Movie {test_movie_id}: {predicted:.2f}")

    # Test 3: User recommendations
    print("\n" + "=" * 70)
    print("TEST 3: USER RECOMMENDATIONS")
    print("=" * 70)

    recommendations = recommender.recommend_for_user(user_id=1, top_n=10)
    if isinstance(recommendations, pd.DataFrame) and len(recommendations) > 0:
        print("\n", recommendations.to_string(index=False))
    else:
        print("No recommendations available")

    print("\n" + "=" * 70)
    print("DONE! Item-based CF ready for production.")
    print("=" * 70)