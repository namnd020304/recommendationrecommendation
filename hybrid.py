import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import warnings


class WeightedHybridRecommender:
    """
    WEIGHTED HYBRID RECOMMENDER

    Kết hợp 3 phương pháp với trọng số:
    1. Content-Based (Genre + Tags)
    2. Item-Based Collaborative Filtering
    3. Popularity/Rating

    Công thức:
    Final_Score = α*Content + β*Collaborative + γ*Popularity
    với α + β + γ = 1
    """

    def __init__(self,
                 content_recommender,
                 collab_recommender,
                 ratings_df: pd.DataFrame,
                 default_weights: Tuple[float, float, float] = (0.4, 0.5, 0.1)):
        """
        Args:
            content_recommender: Instance của HybridContentRecommender
            collab_recommender: Instance của ItemBasedCF
            ratings_df: DataFrame chứa ratings
            default_weights: (content_weight, collab_weight, popularity_weight)
        """
        self.content_rec = content_recommender
        self.collab_rec = collab_recommender
        self.ratings = ratings_df

        # Validate weights
        if not np.isclose(sum(default_weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(default_weights)}")

        self.default_weights = default_weights

        # Precompute popularity scores
        self._compute_popularity_scores()

        print("=" * 70)
        print("WEIGHTED HYBRID RECOMMENDER INITIALIZED")
        print(f"Default weights: Content={default_weights[0]:.2f}, "
              f"Collab={default_weights[1]:.2f}, "
              f"Popularity={default_weights[2]:.2f}")
        print("=" * 70)

    def _compute_popularity_scores(self):
        """
        Tính popularity score cho mỗi phim
        Dựa trên: số ratings và average rating
        """
        print("\nComputing popularity scores...")

        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()

        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']

        # Normalize (min-max scaling)
        movie_stats['norm_rating'] = (
                (movie_stats['avg_rating'] - movie_stats['avg_rating'].min()) /
                (movie_stats['avg_rating'].max() - movie_stats['avg_rating'].min())
        )

        movie_stats['norm_count'] = (
                (movie_stats['rating_count'] - movie_stats['rating_count'].min()) /
                (movie_stats['rating_count'].max() - movie_stats['rating_count'].min())
        )

        # Popularity = 0.6*avg_rating + 0.4*rating_count
        movie_stats['popularity_score'] = (
                0.6 * movie_stats['norm_rating'] +
                0.4 * movie_stats['norm_count']
        )

        # Store as dict for fast lookup
        self.popularity_scores = dict(
            zip(movie_stats['movieId'], movie_stats['popularity_score'])
        )

        print(f"Computed popularity for {len(self.popularity_scores):,} movies")

    def _get_content_scores(self, user_id: int, candidate_movies: List[int]) -> Dict[int, float]:
        """
        Lấy content-based scores cho user
        """
        try:
            # Get full recommendations
            content_recs = self.content_rec.recommend_for_user(
                user_id,
                top_n=len(candidate_movies) * 2,  # Get more to ensure coverage
                exclude_watched=False,
                verbose=False
            )

            if isinstance(content_recs, str):
                return {}

            # Normalize scores to [0, 1]
            if 'score' in content_recs.columns and len(content_recs) > 0:
                scores = content_recs['score'].values
                if scores.max() > scores.min():
                    normalized = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    normalized = np.ones_like(scores)

                content_recs['norm_score'] = normalized

                # Return dict for candidate movies
                score_dict = dict(zip(content_recs['movieId'], content_recs['norm_score']))
                return {m: score_dict.get(m, 0.0) for m in candidate_movies}

            return {}

        except Exception as e:
            warnings.warn(f"Content scoring failed: {str(e)}")
            return {}

    def _get_collab_scores(self, user_id: int, candidate_movies: List[int]) -> Dict[int, float]:
        """
        Lấy collaborative filtering scores với proper normalization
        """
        try:
            collab_scores = {}

            for movie_id in candidate_movies:
                # Predict rating (range: 0.5 - 5.0)
                pred_rating = self.collab_rec.predict_rating(user_id, movie_id)

                if pred_rating is not None:
                    collab_scores[movie_id] = pred_rating
                else:
                    # Fallback to average rating
                    collab_scores[movie_id] = 3.0

            # ✅ PHẦN MỚI: Normalize ALL scores to [0, 1] range
            if collab_scores:
                scores_array = np.array(list(collab_scores.values()))
                min_score = scores_array.min()
                max_score = scores_array.max()

                if max_score > min_score:
                    # Min-max normalization
                    normalized_scores = {}
                    for movie_id, score in collab_scores.items():
                        normalized_scores[movie_id] = (score - min_score) / (max_score - min_score)
                    return normalized_scores
                else:
                    # All scores are the same
                    return {m: 0.5 for m in collab_scores.keys()}

            return {}

        except Exception as e:
            warnings.warn(f"Collab scoring failed: {str(e)}")
            return {}

    def _get_popularity_scores_for_movies(self, candidate_movies: List[int]) -> Dict[int, float]:
        """
        Lấy popularity scores cho các phim
        """
        return {
            movie_id: self.popularity_scores.get(movie_id, 0.0)
            for movie_id in candidate_movies
        }

    def recommend_for_user(self,
                           user_id: int,
                           top_n: int = 10,
                           weights: Optional[Tuple[float, float, float]] = None,
                           exclude_watched: bool = True,
                           strategy: str = 'weighted') -> pd.DataFrame:
        """
        Hybrid recommendation cho user

        Args:
            user_id: User ID
            top_n: Số phim gợi ý
            weights: (content_weight, collab_weight, popularity_weight)
                     Nếu None, dùng default_weights
            exclude_watched: Loại bỏ phim đã xem
            strategy: 'weighted', 'adaptive', 'switching'

        Returns:
            DataFrame với recommendations
        """
        print(f"\n{'=' * 70}")
        print(f"HYBRID RECOMMENDATIONS FOR USER {user_id}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'=' * 70}")

        # Determine weights
        if weights is None:
            if strategy == 'adaptive':
                weights = self._adaptive_weights(user_id)
            else:
                weights = self.default_weights

        # Validate weights
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        w_content, w_collab, w_pop = weights

        print(f"Weights: Content={w_content:.2f}, Collab={w_collab:.2f}, Popularity={w_pop:.2f}")

        # Get watched movies
        if exclude_watched:
            watched_movies = set(
                self.ratings[self.ratings['userId'] == user_id]['movieId'].values
            )
        else:
            watched_movies = set()

        # Get candidate movies (all unwatched movies)
        all_movies = set(self.ratings['movieId'].unique())
        candidate_movies = list(all_movies - watched_movies)

        if len(candidate_movies) == 0:
            return pd.DataFrame(columns=['movieId', 'title', 'final_score', 'rank'])

        print(f"Evaluating {len(candidate_movies):,} candidate movies...")

        # === SWITCHING STRATEGY ===
        if strategy == 'switching':
            return self._switching_strategy(user_id, candidate_movies, top_n, watched_movies)

        # === WEIGHTED STRATEGY ===
        # Get scores from all methods
        print("Computing content-based scores...")
        content_scores = self._get_content_scores(user_id, candidate_movies)

        print("Computing collaborative scores...")
        collab_scores = self._get_collab_scores(user_id, candidate_movies)

        print("Getting popularity scores...")
        pop_scores = self._get_popularity_scores_for_movies(candidate_movies)

        # Combine scores
        final_scores = {}

        for movie_id in candidate_movies:
            content_score = content_scores.get(movie_id, 0.0)
            collab_score = collab_scores.get(movie_id, 0.0)
            pop_score = pop_scores.get(movie_id, 0.0)

            # Weighted sum
            final_score = (
                    w_content * content_score +
                    w_collab * collab_score +
                    w_pop * pop_score
            )

            final_scores[movie_id] = {
                'final_score': final_score,
                'content_score': content_score,
                'collab_score': collab_score,
                'pop_score': pop_score
            }

        # Sort and get top N
        sorted_movies = sorted(
            final_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:top_n]

        # Create results DataFrame
        results = []
        for rank, (movie_id, scores) in enumerate(sorted_movies, 1):
            movie_info = self.content_rec.movies[
                self.content_rec.movies['movieId'] == movie_id
                ]

            if len(movie_info) > 0:
                results.append({
                    'rank': rank,
                    'movieId': movie_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'final_score': scores['final_score'],
                    'content': scores['content_score'],
                    'collab': scores['collab_score'],
                    'popularity': scores['pop_score']
                })

        results_df = pd.DataFrame(results)

        print(f"\nGenerated {len(results_df)} recommendations")
        return results_df

    def _adaptive_weights(self, user_id: int) -> Tuple[float, float, float]:
        """
        Tính trọng số động dựa trên:
        1. Số lượng ratings của user (cold start)
        2. Diversity của ratings
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        n_ratings = len(user_ratings)

        if n_ratings == 0:
            # Cold start: dùng popularity
            return (0.2, 0.2, 0.6)

        elif n_ratings < 10:
            # Ít ratings: ưu tiên content + popularity
            return (0.5, 0.3, 0.2)

        elif n_ratings < 50:
            # Vừa: cân bằng content và collab
            return (0.4, 0.5, 0.1)

        else:
            # Nhiều ratings: ưu tiên collaborative
            return (0.3, 0.6, 0.1)

    def _switching_strategy(self,
                            user_id: int,
                            candidate_movies: List[int],
                            top_n: int,
                            watched_movies: set) -> pd.DataFrame:
        """
        SWITCHING: Chọn method tốt nhất cho từng trường hợp

        Logic:
        - Cold start user (< 5 ratings): Dùng Content-Based + Popularity
        - Normal user: Dùng Collaborative Filtering
        - Cold start movie: Dùng Content-Based
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        n_ratings = len(user_ratings)

        print(f"User has {n_ratings} ratings")

        if n_ratings < 5:
            # Cold start user: Content-Based
            print("→ Using CONTENT-BASED (cold start user)")

            content_recs = self.content_rec.recommend_for_user(
                user_id,
                top_n=top_n,
                exclude_watched=True,
                verbose=False
            )

            if isinstance(content_recs, pd.DataFrame):
                content_recs['method'] = 'content'
                return content_recs
            else:
                return pd.DataFrame()

        else:
            # Normal user: Collaborative
            print("→ Using COLLABORATIVE FILTERING")

            collab_recs = self.collab_rec.recommend_for_user(
                user_id,
                top_n=top_n,
                exclude_watched=True
            )

            if isinstance(collab_recs, pd.DataFrame):
                collab_recs['method'] = 'collaborative'
                return collab_recs
            else:
                return pd.DataFrame()

    def evaluate_weights(self,
                         test_users: List[int],
                         weight_combinations: List[Tuple[float, float, float]],
                         metric: str = 'diversity') -> pd.DataFrame:
        """
        Đánh giá các tổ hợp trọng số khác nhau

        Args:
            test_users: Danh sách user IDs để test
            weight_combinations: List of (w_content, w_collab, w_pop)
            metric: 'diversity' hoặc 'coverage'

        Returns:
            DataFrame với kết quả đánh giá
        """
        results = []

        for weights in weight_combinations:
            print(f"\nTesting weights: {weights}")

            all_recs = []
            unique_movies = set()

            for user_id in test_users:
                try:
                    recs = self.recommend_for_user(
                        user_id,
                        top_n=10,
                        weights=weights,
                        exclude_watched=True,
                        strategy='weighted'
                    )

                    if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                        all_recs.append(recs)
                        unique_movies.update(recs['movieId'].values)

                except Exception as e:
                    continue

            # Calculate metrics
            if metric == 'diversity':
                score = len(unique_movies) / (len(test_users) * 10)  # Diversity ratio
            elif metric == 'coverage':
                total_movies = len(self.ratings['movieId'].unique())
                score = len(unique_movies) / total_movies
            else:
                score = 0.0

            results.append({
                'content_weight': weights[0],
                'collab_weight': weights[1],
                'pop_weight': weights[2],
                'unique_movies': len(unique_movies),
                'metric_score': score
            })

        return pd.DataFrame(results).sort_values('metric_score', ascending=False)


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    """
    Cách sử dụng:

    1. Chuẩn bị các recommenders:
       - content_rec = HybridContentRecommender(...)
       - collab_rec = ItemBasedCF(...)

    2. Khởi tạo hybrid:
       hybrid = WeightedHybridRecommender(
           content_rec, 
           collab_rec,
           ratings_df
       )

    3. Recommend với custom weights:
       recs = hybrid.recommend_for_user(
           user_id=1,
           top_n=10,
           weights=(0.3, 0.6, 0.1)  # content, collab, popularity
       )

    4. Adaptive weights (tự động):
       recs = hybrid.recommend_for_user(
           user_id=1,
           strategy='adaptive'
       )

    5. Switching strategy:
       recs = hybrid.recommend_for_user(
           user_id=1,
           strategy='switching'
       )

    6. Evaluate weights:
       results = hybrid.evaluate_weights(
           test_users=[1, 2, 3, 4, 5],
           weight_combinations=[
               (0.5, 0.4, 0.1),
               (0.3, 0.6, 0.1),
               (0.4, 0.5, 0.1)
           ]
       )
    """

    print(__doc__)