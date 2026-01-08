
import os
import sys
import time
import pickle
import pandas as pd
from typing import Optional, Dict, Any
import warnings

# Import các modules
from item import ItemBasedCF
from conten import HybridContentRecommender
from hybrid import WeightedHybridRecommender


class HybridSystemManager:
    """
    Quản lý toàn bộ hệ thống recommendation
    - Load/save models
    - Configure parameters
    - Switch between strategies
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to config file (Python dict hoặc JSON)
        """
        self.config = self._load_config(config_path)

        # Models
        self.content_model: Optional[HybridContentRecommender] = None
        self.collab_model: Optional[ItemBasedCF] = None
        self.hybrid_model: Optional[WeightedHybridRecommender] = None

        # Data paths
        self.data_dir = self.config.get('data_dir', 'ml-latest')
        self.model_dir = self.config.get('model_dir', 'models')

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

        print("=" * 70)
        print("HYBRID RECOMMENDATION SYSTEM MANAGER")
        print("=" * 70)
        print(f"Data directory: {self.data_dir}")
        print(f"Model directory: {self.model_dir}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'data_dir': 'ml-latest',
            'model_dir': 'models',

            # Content-Based config
            'content': {
                'enabled': True,
                'model_file': 'content_model.pkl'
            },

            # Collaborative Filtering config
            'collab': {
                'enabled': True,
                'model_file': 'collab_model.pkl',
                'min_common_users': 3,
                'top_k_similar': 50,
                'batch_size': 500
            },

            # Hybrid config
            'hybrid': {
                'default_weights': (0.4, 0.5, 0.1),  # content, collab, popularity
                'strategy': 'weighted'  # 'weighted', 'adaptive', 'switching'
            }
        }

        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            default_config.update(custom_config)

        return default_config

    def build_all_models(self, force_rebuild: bool = False):
        """
        Build tất cả models từ scratch

        Args:
            force_rebuild: Nếu True, rebuild ngay cả khi model files đã tồn tại
        """
        print("\n" + "=" * 70)
        print("BUILDING ALL MODELS")
        print("=" * 70)

        # Build Content-Based
        if self.config['content']['enabled']:
            self._build_content_model(force_rebuild)

        # Build Collaborative Filtering
        if self.config['collab']['enabled']:
            self._build_collab_model(force_rebuild)

        # Build Hybrid
        if self.content_model and self.collab_model:
            self._build_hybrid_model()

        print("\n" + "=" * 70)
        print("ALL MODELS BUILT SUCCESSFULLY!")
        print("=" * 70)

    def _build_content_model(self, force_rebuild: bool = False):
        """Build Content-Based model"""
        model_path = os.path.join(self.model_dir, self.config['content']['model_file'])

        if not force_rebuild and os.path.exists(model_path):
            print("\n[Content] Model file exists. Loading...")
            self.load_content_model()
            return

        print("\n[Content] Building from scratch...")
        start = time.time()

        # Initialize
        self.content_model = HybridContentRecommender(
            movies_path=os.path.join(self.data_dir, 'movies.csv'),
            ratings_path=os.path.join(self.data_dir, 'ratings.csv'),
            tags_path=os.path.join(self.data_dir, 'tags.csv')
        )

        # Build features
        self.content_model.prepare_genre_features()
        self.content_model.prepare_tag_features()

        # Save
        self.content_model.save_model(model_path)

        elapsed = time.time() - start
        print(f"[Content] Build time: {elapsed:.2f}s")

    def _build_collab_model(self, force_rebuild: bool = False):
        """Build Collaborative Filtering model"""
        model_path = os.path.join(self.model_dir, self.config['collab']['model_file'])

        if not force_rebuild and os.path.exists(model_path):
            print("\n[Collab] Model file exists. Loading...")
            self.load_collab_model()
            return

        print("\n[Collab] Building from scratch...")
        start = time.time()

        # Initialize
        collab_config = self.config['collab']
        self.collab_model = ItemBasedCF(
            ratings_path=os.path.join(self.data_dir, 'ratings.csv'),
            min_common_users=collab_config.get('min_common_users', 3),
            top_k_similar=collab_config.get('top_k_similar', 50)
        )

        # Build
        self.collab_model.prepare_data()
        self.collab_model.compute_item_similarity(
            batch_size=collab_config.get('batch_size', 500)
        )

        # Save
        self.collab_model.save_model(model_path)

        elapsed = time.time() - start
        print(f"[Collab] Build time: {elapsed / 60:.1f} minutes")

    def _build_hybrid_model(self):
        """Build Hybrid model (requires content + collab)"""
        print("\n[Hybrid] Initializing...")

        ratings_path = os.path.join(self.data_dir, 'ratings.csv')
        ratings = pd.read_csv(ratings_path)

        hybrid_config = self.config['hybrid']
        self.hybrid_model = WeightedHybridRecommender(
            content_recommender=self.content_model,
            collab_recommender=self.collab_model,
            ratings_df=ratings,
            default_weights=tuple(hybrid_config.get('default_weights', (0.4, 0.5, 0.1)))
        )

        print("[Hybrid] Ready!")

    def load_content_model(self):
        """Load pre-trained Content model"""
        model_path = os.path.join(self.model_dir, self.config['content']['model_file'])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Content model not found: {model_path}")

        print(f"\n[Content] Loading from {model_path}...")
        self.content_model = HybridContentRecommender.load_model(
            filepath=model_path,
            ratings_path=os.path.join(self.data_dir, 'ratings.csv')
        )

    def load_collab_model(self):
        """Load pre-trained Collaborative model"""
        model_path = os.path.join(self.model_dir, self.config['collab']['model_file'])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Collab model not found: {model_path}")

        print(f"\n[Collab] Loading from {model_path}...")
        self.collab_model = ItemBasedCF(
            ratings_path=os.path.join(self.data_dir, 'ratings.csv')
        )
        self.collab_model.load_model(model_path)

        # Need to reload ratings for new predictions
        self.collab_model.ratings = pd.read_csv(
            os.path.join(self.data_dir, 'ratings.csv')
        )

    def load_all_models(self):
        """Load tất cả pre-trained models"""
        print("\n" + "=" * 70)
        print("LOADING ALL MODELS")
        print("=" * 70)

        if self.config['content']['enabled']:
            self.load_content_model()

        if self.config['collab']['enabled']:
            self.load_collab_model()

        if self.content_model and self.collab_model:
            self._build_hybrid_model()

        print("\n" + "=" * 70)
        print("ALL MODELS LOADED!")
        print("=" * 70)

    def recommend(self,
                  user_id: int,
                  method: str = 'hybrid',
                  top_n: int = 10,
                  **kwargs) -> pd.DataFrame:
        """
        Unified recommendation interface

        Args:
            user_id: User ID
            method: 'content', 'collab', 'hybrid'
            top_n: Number of recommendations
            **kwargs: Additional parameters for specific methods

        Returns:
            DataFrame with recommendations
        """
        if method == 'content':
            if self.content_model is None:
                raise ValueError("Content model not loaded")
            return self.content_model.recommend_for_user(user_id, top_n=top_n, **kwargs)

        elif method == 'collab':
            if self.collab_model is None:
                raise ValueError("Collab model not loaded")
            return self.collab_model.recommend_for_user(user_id, top_n=top_n, **kwargs)

        elif method == 'hybrid':
            if self.hybrid_model is None:
                raise ValueError("Hybrid model not loaded")
            return self.hybrid_model.recommend_for_user(user_id, top_n=top_n, **kwargs)

        else:
            raise ValueError(f"Unknown method: {method}")

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration và rebuild models nếu cần
        """
        print("\nUpdating configuration...")

        # Check what changed
        rebuild_content = False
        rebuild_collab = False

        if 'collab' in new_config:
            collab_params = ['min_common_users', 'top_k_similar', 'batch_size']
            for param in collab_params:
                if param in new_config['collab']:
                    if new_config['collab'][param] != self.config['collab'].get(param):
                        rebuild_collab = True
                        break

        # Update config
        self.config.update(new_config)

        # Rebuild if needed
        if rebuild_content:
            print("Content parameters changed. Rebuilding...")
            self._build_content_model(force_rebuild=True)

        if rebuild_collab:
            print("Collab parameters changed. Rebuilding...")
            self._build_collab_model(force_rebuild=True)

        # Rebuild hybrid if models changed
        if (rebuild_content or rebuild_collab) and self.content_model and self.collab_model:
            self._build_hybrid_model()

        print("Configuration updated!")

    def compare_methods(self, user_id: int, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        So sánh kết quả từ cả 3 methods
        """
        print(f"\n{'=' * 70}")
        print(f"COMPARING METHODS FOR USER {user_id}")
        print(f"{'=' * 70}")

        results = {}

        # Content-Based
        if self.content_model:
            print("\n[1] Content-Based:")
            try:
                results['content'] = self.recommend(user_id, method='content', top_n=top_n)
            except Exception as e:
                print(f"Error: {e}")

        # Collaborative
        if self.collab_model:
            print("\n[2] Collaborative Filtering:")
            try:
                results['collab'] = self.recommend(user_id, method='collab', top_n=top_n)
            except Exception as e:
                print(f"Error: {e}")

        # Hybrid
        if self.hybrid_model:
            print("\n[3] Hybrid (Weighted):")
            try:
                results['hybrid'] = self.recommend(user_id, method='hybrid', top_n=top_n)
            except Exception as e:
                print(f"Error: {e}")

        return results

    def get_system_info(self) -> Dict[str, Any]:
        """Thông tin về hệ thống"""
        info = {
            'models_loaded': {
                'content': self.content_model is not None,
                'collab': self.collab_model is not None,
                'hybrid': self.hybrid_model is not None
            },
            'config': self.config
        }

        if self.content_model:
            info['content_stats'] = {
                'n_movies': len(self.content_model.movies),
                'n_genres': len(self.content_model.genre_cols) if self.content_model.genre_cols else 0
            }

        if self.collab_model:
            info['collab_stats'] = {
                'n_movies': len(self.collab_model.movie_to_idx),
                'n_users': len(self.collab_model.user_to_idx)
            }

        return info


# ===== USAGE EXAMPLES =====
def example_usage():
    """Examples của cách sử dụng"""

    # ===== 1. QUICK START - Load existing models =====
    print("\n" + "=" * 70)
    print("EXAMPLE 1: QUICK START")
    print("=" * 70)

    manager = HybridSystemManager()

    # Load pre-trained models
    try:
        manager.load_all_models()
    except FileNotFoundError:
        print("Models not found. Building from scratch...")
        manager.build_all_models()

    # Get recommendations
    recs = manager.recommend(user_id=1, method='hybrid', top_n=10)
    print("\nHybrid Recommendations:")
    print(recs[['rank', 'title', 'final_score']].to_string(index=False))

    # ===== 2. BUILD FROM SCRATCH =====
    print("\n" + "=" * 70)
    print("EXAMPLE 2: BUILD FROM SCRATCH")
    print("=" * 70)

    manager2 = HybridSystemManager()
    manager2.build_all_models(force_rebuild=True)

    # ===== 3. CUSTOM CONFIGURATION =====
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CUSTOM CONFIG")
    print("=" * 70)

    custom_config = {
        'collab': {
            'enabled': True,
            'min_common_users': 5,  # Stricter filtering
            'top_k_similar': 30,  # Fewer similar items
            'batch_size': 1000  # Larger batches
        },
        'hybrid': {
            'default_weights': (0.3, 0.6, 0.1),  # More weight on collab
            'strategy': 'adaptive'
        }
    }

    manager3 = HybridSystemManager()
    manager3.config.update(custom_config)
    manager3.build_all_models()

    # ===== 4. COMPARE METHODS =====
    print("\n" + "=" * 70)
    print("EXAMPLE 4: COMPARE METHODS")
    print("=" * 70)

    comparison = manager.compare_methods(user_id=1, top_n=5)

    for method, results in comparison.items():
        print(f"\n{method.upper()}:")
        if isinstance(results, pd.DataFrame):
            print(results[['title']].to_string(index=False))

    # ===== 5. DIFFERENT STRATEGIES =====
    print("\n" + "=" * 70)
    print("EXAMPLE 5: DIFFERENT STRATEGIES")
    print("=" * 70)

    # Weighted
    recs_weighted = manager.recommend(
        user_id=1,
        method='hybrid',
        weights=(0.5, 0.4, 0.1),
        strategy='weighted'
    )

    # Adaptive
    recs_adaptive = manager.recommend(
        user_id=1,
        method='hybrid',
        strategy='adaptive'
    )

    # Switching
    recs_switching = manager.recommend(
        user_id=1,
        method='hybrid',
        strategy='switching'
    )

    # ===== 6. UPDATE CONFIGURATION AT RUNTIME =====
    print("\n" + "=" * 70)
    print("EXAMPLE 6: RUNTIME CONFIG UPDATE")
    print("=" * 70)

    new_config = {
        'hybrid': {
            'default_weights': (0.2, 0.7, 0.1)  # Change weights
        }
    }

    manager.update_config(new_config)

    # ===== 7. SYSTEM INFO =====
    print("\n" + "=" * 70)
    print("EXAMPLE 7: SYSTEM INFO")
    print("=" * 70)

    info = manager.get_system_info()
    print(f"\nModels loaded: {info['models_loaded']}")
    if 'content_stats' in info:
        print(f"Content stats: {info['content_stats']}")


# ===== MAIN =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid Recommendation System')
    parser.add_argument('--build', action='store_true', help='Build models from scratch')
    parser.add_argument('--load', action='store_true', help='Load existing models')
    parser.add_argument('--user', type=int, help='Get recommendations for user')
    parser.add_argument('--method', choices=['content', 'collab', 'hybrid'],
                        default='hybrid', help='Recommendation method')
    parser.add_argument('--top-n', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--compare', action='store_true', help='Compare all methods')
    parser.add_argument('--examples', action='store_true', help='Run usage examples')

    args = parser.parse_args()

    if args.examples:
        example_usage()
        sys.exit(0)

    # Initialize manager
    manager = HybridSystemManager()

    if args.build:
        manager.build_all_models(force_rebuild=True)

    elif args.load:
        manager.load_all_models()

    else:
        # Try load, fallback to build
        try:
            manager.load_all_models()
        except FileNotFoundError:
            print("Models not found. Building...")
            manager.build_all_models()

    # Get recommendations
    if args.user:
        if args.compare:
            results = manager.compare_methods(args.user, top_n=args.top_n)
            for method, recs in results.items():
                print(f"\n{method.upper()}:")
                if isinstance(recs, pd.DataFrame):
                    print(recs.to_string(index=False))
        else:
            recs = manager.recommend(args.user, method=args.method, top_n=args.top_n)
            print("\nRecommendations:")
            print(recs.to_string(index=False))