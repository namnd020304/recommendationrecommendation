import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict
import time


class RecommenderEvaluator:
    """
    Module đánh giá hệ thống recommendation với 4 chỉ số chính:
    1. Precision@K & Recall@K
    2. Diversity
    3. Coverage
    4. Novelty
    """

    def __init__(self, manager, ratings_df: pd.DataFrame):
        """
        Args:
            manager: HybridSystemManager instance
            ratings_df: Full ratings DataFrame
        """
        self.manager = manager
        self.ratings = ratings_df

        # Precompute movie popularity for Novelty metric
        self._compute_popularity()

        print("=" * 70)
        print("RECOMMENDER EVALUATOR INITIALIZED")
        print(f"Total ratings: {len(self.ratings):,}")
        print(f"Total users: {self.ratings['userId'].nunique():,}")
        print(f"Total movies: {self.ratings['movieId'].nunique():,}")
        print("=" * 70)

    def _compute_popularity(self):
        """Tính popularity của mỗi phim (cho Novelty metric)"""
        movie_counts = self.ratings['movieId'].value_counts()
        total_ratings = len(self.ratings)

        # Popularity = số lần rating / tổng số ratings
        self.movie_popularity = (movie_counts / total_ratings).to_dict()

        print(f"Computed popularity for {len(self.movie_popularity):,} movies")

    def train_test_split(self,
                         test_size: float = 0.2,
                         min_ratings_per_user: int = 10,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        """
        Split data thành train/test set

        Strategy: Với mỗi user, ẩn 20% ratings có rating >= 4.0 (relevant items)

        Returns:
            train_ratings: Training set
            test_ratings: Test set (hidden ratings)
            test_users: List of user IDs in test set
        """
        print("\n" + "=" * 70)
        print("TRAIN-TEST SPLIT")
        print("=" * 70)

        np.random.seed(random_state)

        # Lọc users có đủ ratings
        user_counts = self.ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_per_user].index

        print(f"Users with >= {min_ratings_per_user} ratings: {len(valid_users):,}")

        train_list = []
        test_list = []
        test_users = []

        for user_id in valid_users:
            user_ratings = self.ratings[self.ratings['userId'] == user_id].copy()

            # Chỉ split relevant items (rating >= 4.0)
            relevant = user_ratings[user_ratings['rating'] >= 4.0]
            non_relevant = user_ratings[user_ratings['rating'] < 4.0]

            if len(relevant) < 2:
                # Không đủ relevant items để split
                train_list.append(user_ratings)
                continue

            # Split relevant items
            n_test = max(1, int(len(relevant) * test_size))
            test_indices = np.random.choice(relevant.index, size=n_test, replace=False)

            test_items = user_ratings.loc[test_indices]
            train_items = user_ratings.drop(test_indices)

            train_list.append(train_items)
            test_list.append(test_items)
            test_users.append(user_id)

        train_ratings = pd.concat(train_list, ignore_index=True)
        test_ratings = pd.concat(test_list, ignore_index=True)

        print(f"\nTrain set: {len(train_ratings):,} ratings")
        print(f"Test set: {len(test_ratings):,} ratings")
        print(f"Test users: {len(test_users):,}")

        return train_ratings, test_ratings, test_users

    def precision_recall_at_k(self,
                              test_users: List[int],
                              test_ratings: pd.DataFrame,
                              method: str = 'hybrid',
                              k: int = 10,
                              relevance_threshold: float = 4.0) -> Dict[str, float]:
        """
        Tính Precision@K và Recall@K

        Args:
            test_users: List user IDs để test
            test_ratings: Hidden ratings (ground truth)
            method: 'content', 'collab', 'hybrid'
            k: Top K recommendations
            relevance_threshold: Rating >= threshold = relevant

        Returns:
            Dict với precision, recall, f1
        """
        print("\n" + "=" * 70)
        print(f"PRECISION & RECALL @ {k} - Method: {method.upper()}")
        print("=" * 70)

        precisions = []
        recalls = []

        successful_users = 0
        failed_users = 0

        for user_id in test_users:
            # Ground truth: relevant items trong test set
            user_test = test_ratings[test_ratings['userId'] == user_id]
            relevant_items = set(
                user_test[user_test['rating'] >= relevance_threshold]['movieId'].values
            )

            if len(relevant_items) == 0:
                continue

            # Get recommendations
            try:
                recs = self.manager.recommend(
                    user_id,
                    method=method,
                    top_n=k,
                    exclude_watched=True
                )

                if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                    recommended_items = set(recs['movieId'].values[:k])

                    # True Positives
                    hits = recommended_items & relevant_items

                    # Precision@K
                    precision = len(hits) / k if k > 0 else 0

                    # Recall@K
                    recall = len(hits) / len(relevant_items) if len(relevant_items) > 0 else 0

                    precisions.append(precision)
                    recalls.append(recall)
                    successful_users += 1
                else:
                    failed_users += 1

            except Exception as e:
                failed_users += 1
                continue

        # Aggregate metrics
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0

        # F1 Score
        if avg_precision + avg_recall > 0:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1 = 0

        results = {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1,
            'evaluated_users': successful_users,
            'failed_users': failed_users
        }

        print(f"\nResults:")
        print(f"  Precision@{k}: {avg_precision:.4f}")
        print(f"  Recall@{k}: {avg_recall:.4f}")
        print(f"  F1@{k}: {f1:.4f}")
        print(f"  Evaluated users: {successful_users}/{len(test_users)}")

        return results

    def diversity_metrics(self,
                         test_users: List[int],
                         method: str = 'hybrid',
                         k: int = 10) -> Dict[str, float]:
        """
        Tính Diversity metrics

        1. Intra-List Diversity: Đa dạng trong 1 list (avg pairwise distance)
        2. Genre Diversity: Số genres unique
        3. Content Diversity: Dựa trên content features

        Returns:
            Dict với các diversity metrics
        """
        print("\n" + "=" * 70)
        print(f"DIVERSITY METRICS @ {k} - Method: {method.upper()}")
        print("=" * 70)

        intra_list_diversities = []
        genre_diversities = []
        all_genres_count = []

        for user_id in test_users:
            try:
                recs = self.manager.recommend(
                    user_id,
                    method=method,
                    top_n=k,
                    exclude_watched=True
                )

                if not isinstance(recs, pd.DataFrame) or len(recs) < 2:
                    continue

                movie_ids = recs['movieId'].values[:k]

                # Genre Diversity
                genres_set = set()
                genre_lists = []

                for mid in movie_ids:
                    movie_info = self.manager.content_model.movies[
                        self.manager.content_model.movies['movieId'] == mid
                    ]

                    if len(movie_info) > 0:
                        genres = movie_info.iloc[0]['genres'].split('|')
                        genres_set.update(genres)
                        genre_lists.append(set(genres))

                genre_diversities.append(len(genres_set))

                # Intra-List Diversity (genre-based)
                # Tính Jaccard distance trung bình giữa các cặp phim
                pairwise_distances = []
                for i in range(len(genre_lists)):
                    for j in range(i + 1, len(genre_lists)):
                        # Jaccard distance = 1 - Jaccard similarity
                        intersection = len(genre_lists[i] & genre_lists[j])
                        union = len(genre_lists[i] | genre_lists[j])

                        if union > 0:
                            jaccard_sim = intersection / union
                            jaccard_dist = 1 - jaccard_sim
                            pairwise_distances.append(jaccard_dist)

                if pairwise_distances:
                    intra_list_diversities.append(np.mean(pairwise_distances))

            except Exception as e:
                continue

        # Aggregate
        results = {
            'intra_list_diversity': np.mean(intra_list_diversities) if intra_list_diversities else 0,
            'avg_genre_count': np.mean(genre_diversities) if genre_diversities else 0,
            'evaluated_users': len(intra_list_diversities)
        }

        print(f"\nResults:")
        print(f"  Intra-List Diversity: {results['intra_list_diversity']:.4f}")
        print(f"  Avg Genre Count: {results['avg_genre_count']:.2f}")
        print(f"  Evaluated users: {results['evaluated_users']}")

        return results

    def coverage_metrics(self,
                        test_users: List[int],
                        method: str = 'hybrid',
                        k: int = 10) -> Dict[str, float]:
        """
        Tính Coverage metrics

        1. Catalog Coverage: % phim được gợi ý
        2. User Coverage: % users nhận được đủ K recommendations

        Returns:
            Dict với coverage metrics
        """
        print("\n" + "=" * 70)
        print(f"COVERAGE METRICS @ {k} - Method: {method.upper()}")
        print("=" * 70)

        all_recommended_movies = set()
        users_with_k_recs = 0

        for user_id in test_users:
            try:
                recs = self.manager.recommend(
                    user_id,
                    method=method,
                    top_n=k,
                    exclude_watched=True
                )

                if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                    recommended = recs['movieId'].values[:k]
                    all_recommended_movies.update(recommended)

                    if len(recommended) >= k:
                        users_with_k_recs += 1

            except Exception as e:
                continue

        # Total movies in catalog
        total_movies = self.ratings['movieId'].nunique()

        # Metrics
        catalog_coverage = len(all_recommended_movies) / total_movies
        user_coverage = users_with_k_recs / len(test_users)

        results = {
            'catalog_coverage': catalog_coverage,
            'user_coverage': user_coverage,
            'unique_movies_recommended': len(all_recommended_movies),
            'total_movies': total_movies
        }

        print(f"\nResults:")
        print(f"  Catalog Coverage: {catalog_coverage:.4f} ({len(all_recommended_movies):,}/{total_movies:,})")
        print(f"  User Coverage: {user_coverage:.4f} ({users_with_k_recs}/{len(test_users)})")

        return results

    def novelty_metric(self,
                      test_users: List[int],
                      method: str = 'hybrid',
                      k: int = 10) -> Dict[str, float]:
        """
        Tính Novelty: Khả năng gợi ý phim ít phổ biến

        Novelty = -log2(popularity) trung bình

        Popularity cao → Novelty thấp (phim phổ biến)
        Popularity thấp → Novelty cao (phim niche)

        Returns:
            Dict với novelty metrics
        """
        print("\n" + "=" * 70)
        print(f"NOVELTY METRIC @ {k} - Method: {method.upper()}")
        print("=" * 70)

        novelties = []

        for user_id in test_users:
            try:
                recs = self.manager.recommend(
                    user_id,
                    method=method,
                    top_n=k,
                    exclude_watched=True
                )

                if not isinstance(recs, pd.DataFrame) or len(recs) == 0:
                    continue

                movie_ids = recs['movieId'].values[:k]

                # Tính novelty cho mỗi phim
                user_novelties = []
                for mid in movie_ids:
                    if mid in self.movie_popularity:
                        pop = self.movie_popularity[mid]
                        # Novelty = -log2(popularity)
                        # Thêm epsilon để tránh log(0)
                        novelty = -np.log2(pop + 1e-10)
                        user_novelties.append(novelty)

                if user_novelties:
                    novelties.append(np.mean(user_novelties))

            except Exception as e:
                continue

        avg_novelty = np.mean(novelties) if novelties else 0

        results = {
            'novelty': avg_novelty,
            'evaluated_users': len(novelties)
        }

        print(f"\nResults:")
        print(f"  Novelty: {avg_novelty:.4f} (higher = more novel/niche)")
        print(f"  Evaluated users: {len(novelties)}")

        return results

    def comprehensive_evaluation(self,
                                test_users: List[int],
                                test_ratings: pd.DataFrame,
                                methods: List[str] = ['content', 'collab', 'hybrid'],
                                k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Đánh giá toàn diện với nhiều methods và k values

        Returns:
            DataFrame với kết quả so sánh
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 70)

        results = []

        for method in methods:
            print(f"\n{'=' * 70}")
            print(f"EVALUATING METHOD: {method.upper()}")
            print(f"{'=' * 70}")

            for k in k_values:
                print(f"\n--- K = {k} ---")

                start_time = time.time()

                # 1. Precision & Recall
                pr_results = self.precision_recall_at_k(
                    test_users, test_ratings, method=method, k=k
                )

                # 2. Diversity
                div_results = self.diversity_metrics(
                    test_users, method=method, k=k
                )

                # 3. Coverage
                cov_results = self.coverage_metrics(
                    test_users, method=method, k=k
                )

                # 4. Novelty
                nov_results = self.novelty_metric(
                    test_users, method=method, k=k
                )

                elapsed = time.time() - start_time

                # Combine results
                row = {
                    'method': method,
                    'k': k,
                    'precision': pr_results['precision@k'],
                    'recall': pr_results['recall@k'],
                    'f1': pr_results['f1@k'],
                    'diversity': div_results['intra_list_diversity'],
                    'genre_count': div_results['avg_genre_count'],
                    'catalog_coverage': cov_results['catalog_coverage'],
                    'user_coverage': cov_results['user_coverage'],
                    'novelty': nov_results['novelty'],
                    'eval_time_sec': elapsed
                }

                results.append(row)

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(results_df.to_string(index=False))

        return results_df

    def compare_strategies(self,
                          test_users: List[int],
                          test_ratings: pd.DataFrame,
                          strategies: List[str] = ['weighted', 'adaptive', 'switching'],
                          k: int = 10) -> pd.DataFrame:
        """
        So sánh các hybrid strategies

        Returns:
            DataFrame với kết quả
        """
        print("\n" + "=" * 70)
        print(f"COMPARING HYBRID STRATEGIES @ K={k}")
        print("=" * 70)

        results = []

        for strategy in strategies:
            print(f"\n{'=' * 70}")
            print(f"STRATEGY: {strategy.upper()}")
            print(f"{'=' * 70}")

            # Precision & Recall
            pr_results = self.precision_recall_at_k(
                test_users, test_ratings, method='hybrid', k=k
            )

            # Diversity
            div_results = self.diversity_metrics(
                test_users, method='hybrid', k=k
            )

            # Coverage
            cov_results = self.coverage_metrics(
                test_users, method='hybrid', k=k
            )

            # Novelty
            nov_results = self.novelty_metric(
                test_users, method='hybrid', k=k
            )

            row = {
                'strategy': strategy,
                'precision': pr_results['precision@k'],
                'recall': pr_results['recall@k'],
                'f1': pr_results['f1@k'],
                'diversity': div_results['intra_list_diversity'],
                'coverage': cov_results['catalog_coverage'],
                'novelty': nov_results['novelty']
            }

            results.append(row)

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        print(results_df.to_string(index=False))

        return results_df

    def save_evaluation_results(self, results_df: pd.DataFrame, filepath: str):
        """Save evaluation results to CSV"""
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")

    def comprehensive_evaluation_optimized(self,
                                           test_users: List[int],
                                           test_ratings: pd.DataFrame,
                                           methods: List[str] = ['content', 'collab', 'hybrid'],
                                           k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        OPTIMIZED: Cache recommendations để tránh tính lại
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE EVALUATION (OPTIMIZED)")
        print("=" * 70)

        results = []
        max_k = max(k_values)

        for method in methods:
            print(f"\n{'=' * 70}")
            print(f"EVALUATING METHOD: {method.upper()}")
            print(f"{'=' * 70}")

            # ✅ Pre-generate tất cả recommendations 1 lần
            print(f"Pre-generating recommendations for {len(test_users)} users...")
            recommendations_cache = {}

            for idx, user_id in enumerate(test_users):
                if idx % 20 == 0:
                    print(f"  Progress: {idx}/{len(test_users)}")

                try:
                    recs = self.manager.recommend(
                        user_id,
                        method=method,
                        top_n=max_k,
                        exclude_watched=True
                    )
                    recommendations_cache[user_id] = recs
                except Exception as e:
                    recommendations_cache[user_id] = pd.DataFrame()

            print(f"✅ Cached {len(recommendations_cache)} recommendations")

            # Evaluate cho từng k value
            for k in k_values:
                print(f"\n--- K = {k} ---")
                start_time = time.time()

                # ✅ Sử dụng cache
                pr_results = self._precision_recall_cached(
                    test_users, test_ratings, recommendations_cache, k
                )

                div_results = self._diversity_cached(
                    test_users, recommendations_cache, k
                )

                cov_results = self._coverage_cached(
                    test_users, recommendations_cache, k
                )

                nov_results = self._novelty_cached(
                    test_users, recommendations_cache, k
                )

                elapsed = time.time() - start_time

                row = {
                    'method': method,
                    'k': k,
                    'precision': pr_results['precision@k'],
                    'recall': pr_results['recall@k'],
                    'f1': pr_results['f1@k'],
                    'diversity': div_results['intra_list_diversity'],
                    'genre_count': div_results['avg_genre_count'],
                    'catalog_coverage': cov_results['catalog_coverage'],
                    'user_coverage': cov_results['user_coverage'],
                    'novelty': nov_results['novelty'],
                    'eval_time_sec': elapsed
                }

                results.append(row)

        results_df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(results_df.to_string(index=False))

        return results_df

    def _precision_recall_cached(self, test_users, test_ratings, recommendations_cache, k):
        precisions = []
        recalls = []
        successful_users = 0

        for user_id in test_users:
            user_test = test_ratings[test_ratings['userId'] == user_id]
            relevant_items = set(user_test[user_test['rating'] >= 4.0]['movieId'].values)

            if len(relevant_items) == 0:
                continue

            recs = recommendations_cache.get(user_id, pd.DataFrame())

            if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                recommended_items = set(recs['movieId'].values[:k])
                hits = recommended_items & relevant_items

                precision = len(hits) / k if k > 0 else 0
                recall = len(hits) / len(relevant_items) if len(relevant_items) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)
                successful_users += 1

        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1,
            'evaluated_users': successful_users
        }

    def _diversity_cached(self, test_users, recommendations_cache, k):
        intra_list_diversities = []
        genre_diversities = []

        for user_id in test_users:
            recs = recommendations_cache.get(user_id, pd.DataFrame())

            if not isinstance(recs, pd.DataFrame) or len(recs) < 2:
                continue

            movie_ids = recs['movieId'].values[:k]
            genres_set = set()
            genre_lists = []

            for mid in movie_ids:
                movie_info = self.manager.content_model.movies[
                    self.manager.content_model.movies['movieId'] == mid
                    ]

                if len(movie_info) > 0:
                    genres = movie_info.iloc[0]['genres'].split('|')
                    genres_set.update(genres)
                    genre_lists.append(set(genres))

            genre_diversities.append(len(genres_set))

            pairwise_distances = []
            for i in range(len(genre_lists)):
                for j in range(i + 1, len(genre_lists)):
                    intersection = len(genre_lists[i] & genre_lists[j])
                    union = len(genre_lists[i] | genre_lists[j])

                    if union > 0:
                        jaccard_dist = 1 - (intersection / union)
                        pairwise_distances.append(jaccard_dist)

            if pairwise_distances:
                intra_list_diversities.append(np.mean(pairwise_distances))

        return {
            'intra_list_diversity': np.mean(intra_list_diversities) if intra_list_diversities else 0,
            'avg_genre_count': np.mean(genre_diversities) if genre_diversities else 0,
            'evaluated_users': len(intra_list_diversities)
        }

    def _coverage_cached(self, test_users, recommendations_cache, k):
        all_recommended_movies = set()
        users_with_k_recs = 0

        for user_id in test_users:
            recs = recommendations_cache.get(user_id, pd.DataFrame())

            if isinstance(recs, pd.DataFrame) and len(recs) > 0:
                recommended = recs['movieId'].values[:k]
                all_recommended_movies.update(recommended)

                if len(recommended) >= k:
                    users_with_k_recs += 1

        total_movies = self.ratings['movieId'].nunique()

        return {
            'catalog_coverage': len(all_recommended_movies) / total_movies,
            'user_coverage': users_with_k_recs / len(test_users),
            'unique_movies_recommended': len(all_recommended_movies),
            'total_movies': total_movies
        }

    def _novelty_cached(self, test_users, recommendations_cache, k):
        novelties = []

        for user_id in test_users:
            recs = recommendations_cache.get(user_id, pd.DataFrame())

            if not isinstance(recs, pd.DataFrame) or len(recs) == 0:
                continue

            movie_ids = recs['movieId'].values[:k]
            user_novelties = []

            for mid in movie_ids:
                if mid in self.movie_popularity:
                    pop = self.movie_popularity[mid]
                    novelty = -np.log2(pop + 1e-10)
                    user_novelties.append(novelty)

            if user_novelties:
                novelties.append(np.mean(user_novelties))

        return {
            'novelty': np.mean(novelties) if novelties else 0,
            'evaluated_users': len(novelties)
        }

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    """
    Cách sử dụng:
    
    1. Tạo evaluator:
       evaluator = RecommenderEvaluator(manager, ratings_df)
    
    2. Train-test split:
       train, test, test_users = evaluator.train_test_split(test_size=0.2)
    
    3. Rebuild models với train data (optional)
    
    4. Evaluate:
       results = evaluator.comprehensive_evaluation(
           test_users, test, 
           methods=['content', 'collab', 'hybrid'],
           k_values=[5, 10, 20]
       )
    
    5. Compare strategies:
       strategy_results = evaluator.compare_strategies(
           test_users, test,
           strategies=['weighted', 'adaptive', 'switching']
       )
    
    6. Save:
       evaluator.save_evaluation_results(results, 'evaluation_results.csv')
    """

    print(__doc__)