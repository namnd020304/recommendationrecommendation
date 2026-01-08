"""
CONFIGURATION FILE
Centralized configuration cho toàn bộ hệ thống
"""

import os

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ml-latest')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data files
MOVIES_FILE = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_FILE = os.path.join(DATA_DIR, 'ratings.csv')
TAGS_FILE = os.path.join(DATA_DIR, 'tags.csv')

# ===== CONTENT-BASED CONFIGURATION =====
CONTENT_CONFIG = {
    'enabled': True,
    'model_file': 'content_model.pkl',

    # Genre-based parameters
    'genre_weight': 0.5,  # Initial weight (will be adaptive)

    # Tag-based TF-IDF parameters
    'tfidf': {
        'max_features': 3000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.7,
    },

    # Tag filtering
    'min_tag_frequency': 3,  # Tags phải xuất hiện >= 3 lần

    # Caching
    'cache_max_size': 1000,
    'use_cache': True,
}

# ===== COLLABORATIVE FILTERING CONFIGURATION =====
COLLAB_CONFIG = {
    'enabled': True,
    'model_file': 'collab_model.pkl',

    # Data filtering
    'min_user_ratings': 5,  # User phải có ít nhất 5 ratings
    'min_movie_ratings': 3,  # Movie phải có ít nhất 3 ratings

    # Similarity computation
    'min_common_users': 3,  # Minimum users chung để tính similarity
    'top_k_similar': 50,  # Lưu top K phim tương tự

    # Performance
    'batch_size': 500,  # Batch size khi tính similarity

    # Prediction
    'default_rating': 3.0,  # Default rating cho cold start
    'rating_range': (0.5, 5.0),  # Clamp predictions
}

# ===== HYBRID SYSTEM CONFIGURATION =====
HYBRID_CONFIG = {
    'enabled': True,

    # Default strategy
    'default_strategy': 'adaptive',  # 'weighted', 'adaptive', 'switching'

    # Default weights (content, collab, popularity)
    'default_weights': {
        'content': 0.4,
        'collab': 0.5,
        'popularity': 0.1
    },

    # Adaptive weights based on user profile
    'adaptive_weights': {
        'cold_start': {  # < 5 ratings
            'content': 0.2,
            'collab': 0.2,
            'popularity': 0.6
        },
        'few_ratings': {  # 5-10 ratings
            'content': 0.5,
            'collab': 0.3,
            'popularity': 0.2
        },
        'medium_ratings': {  # 10-50 ratings
            'content': 0.4,
            'collab': 0.5,
            'popularity': 0.1
        },
        'many_ratings': {  # 50+ ratings
            'content': 0.3,
            'collab': 0.6,
            'popularity': 0.1
        }
    },

    # Popularity scoring
    'popularity': {
        'rating_weight': 0.6,  # Weight for avg rating
        'count_weight': 0.4,  # Weight for rating count
    },

    # Switching strategy thresholds
    'switching': {
        'cold_start_threshold': 5,  # < 5 ratings → use content
        'use_collab_threshold': 5,  # >= 5 ratings → use collab
    }
}

# ===== EVALUATION CONFIGURATION =====
EVALUATION_CONFIG = {
    # Test users
    'n_test_users': 100,
    'test_user_min_ratings': 10,

    # Metrics
    'metrics': ['precision', 'recall', 'diversity', 'coverage'],

    # Cross-validation
    'n_folds': 5,
    'test_size': 0.2,
}

# ===== PERFORMANCE CONFIGURATION =====
PERFORMANCE_CONFIG = {
    # Memory management
    'gc_after_build': True,

    # Parallel processing
    'n_jobs': -1,  # Use all CPUs

    # Logging
    'verbose': True,
    'log_file': os.path.join(MODEL_DIR, 'training.log'),
}

# ===== PRODUCTION CONFIGURATION =====
PRODUCTION_CONFIG = {
    # API settings
    'max_recommendations': 50,
    'default_recommendations': 10,

    # Caching
    'cache_recommendations': True,
    'cache_ttl': 3600,  # 1 hour

    # Rate limiting
    'rate_limit_enabled': True,
    'max_requests_per_minute': 60,

    # Model refresh
    'auto_retrain': False,
    'retrain_interval_days': 7,
}

# ===== EXPERIMENT CONFIGURATIONS =====
# Có thể định nghĩa nhiều experiments khác nhau

EXPERIMENT_CONFIGS = {
    # Experiment 1: Prioritize Content
    'content_heavy': {
        'hybrid': {
            'default_weights': {
                'content': 0.6,
                'collab': 0.3,
                'popularity': 0.1
            }
        }
    },

    # Experiment 2: Prioritize Collaborative
    'collab_heavy': {
        'hybrid': {
            'default_weights': {
                'content': 0.3,
                'collab': 0.6,
                'popularity': 0.1
            }
        }
    },

    # Experiment 3: More similar items
    'high_similarity': {
        'collab': {
            'top_k_similar': 100,
            'min_common_users': 5
        }
    },

    # Experiment 4: Faster training
    'fast_training': {
        'collab': {
            'batch_size': 1000,
            'top_k_similar': 30
        },
        'content': {
            'tfidf': {
                'max_features': 1500
            }
        }
    }
}


# ===== HELPER FUNCTIONS =====

def get_full_config():
    """Merge tất cả configs thành 1 dict"""
    return {
        'paths': {
            'data_dir': DATA_DIR,
            'model_dir': MODEL_DIR,
            'movies_file': MOVIES_FILE,
            'ratings_file': RATINGS_FILE,
            'tags_file': TAGS_FILE
        },
        'content': CONTENT_CONFIG,
        'collab': COLLAB_CONFIG,
        'hybrid': HYBRID_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'production': PRODUCTION_CONFIG
    }


def get_experiment_config(experiment_name: str):
    """
    Lấy config cho một experiment cụ thể

    Args:
        experiment_name: Tên experiment trong EXPERIMENT_CONFIGS

    Returns:
        Merged config
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    base_config = get_full_config()
    experiment_config = EXPERIMENT_CONFIGS[experiment_name]

    # Deep merge
    for key, value in experiment_config.items():
        if key in base_config and isinstance(value, dict):
            base_config[key].update(value)
        else:
            base_config[key] = value

    return base_config


def save_config_to_json(filepath: str, config: dict = None):
    """Save config to JSON file"""
    import json

    if config is None:
        config = get_full_config()

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {filepath}")


def load_config_from_json(filepath: str):
    """Load config from JSON file"""
    import json

    with open(filepath, 'r') as f:
        config = json.load(f)

    return config


def print_current_config():
    """Print current configuration"""
    import json
    config = get_full_config()
    print("=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(json.dumps(config, indent=2))


# ===== VALIDATION =====

def validate_config(config: dict = None):
    """
    Validate configuration

    Checks:
    - File paths exist
    - Weights sum to 1.0
    - Parameters are in valid ranges
    """
    if config is None:
        config = get_full_config()

    errors = []
    warnings = []

    # Check paths
    if not os.path.exists(config['paths']['data_dir']):
        errors.append(f"Data directory not found: {config['paths']['data_dir']}")

    # Check weights
    hybrid_weights = config['hybrid']['default_weights']
    weight_sum = sum(hybrid_weights.values())
    if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point errors
        errors.append(f"Hybrid weights must sum to 1.0, got {weight_sum}")

    # Check collab parameters
    if config['collab']['top_k_similar'] > 200:
        warnings.append("top_k_similar > 200 may consume too much memory")

    if config['collab']['batch_size'] > 2000:
        warnings.append("batch_size > 2000 may be slow")

    # Print results
    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  ❌ {error}")

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    if not errors and not warnings:
        print("✅ Configuration is valid!")

    return len(errors) == 0


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURATION MODULE")
    print("=" * 70)

    # Print current config
    print_current_config()

    # Validate
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    validate_config()

    # Save to JSON
    print("\n" + "=" * 70)
    print("SAVE TO JSON")
    print("=" * 70)
    save_config_to_json('config.json')

    # Get experiment config
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIG")
    print("=" * 70)
    exp_config = get_experiment_config('content_heavy')
    print(f"Content weight: {exp_config['hybrid']['default_weights']['content']}")