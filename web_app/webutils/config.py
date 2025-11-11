
"""
Configuration settings for AutoG-S Web App
Based on the actual AutoG-S backend pipeline
"""

# LLM Configuration Constants
CONTEXT_SIZE = 65536
OUTPUT_SIZE = 65536

# LLM Model Configurations
LLM_MODELS = {
    "sonnet4": {
        "name": "Claude Sonnet 4",
        "model_id": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "description": "Latest Claude Sonnet 4 model for advanced reasoning",
        "context_size": CONTEXT_SIZE,
        "output_size": OUTPUT_SIZE
    },
    "llama3": {
        "name": "Llama 3 70B",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "description": "Open-source model with strong performance",
        "context_size": CONTEXT_SIZE,
        "output_size": OUTPUT_SIZE
    },
    "mistralarge": {
        "name": "Mistral Large",
        "model_id": "mistral.mistral-large-2402-v1:0",
        "description": "High-performance multilingual model",
        "context_size": CONTEXT_SIZE,
        "output_size": OUTPUT_SIZE
    },
    "sonnet45": {
        "name": "Claude Sonnet 4.5",
        "model_id": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "description": "Latest Claude Sonnet 4.5 with enhanced performance",
        "context_size": CONTEXT_SIZE,
        "output_size": OUTPUT_SIZE
    }
}

# AutoG Configuration (matching actual task definitions)
AUTOG_CONFIG = {
    "methods": [
        "autog-s"  # AutoG-S directly adopts the final output state (only method supported in web app)
    ],
    "datasets": {
        "custom": [
            "relation", # Find primary/foreign keys
            "kg",       # Knowledge graph construction
            "kg2"       # Enhanced knowledge graph construction
        ],
        "mag": [
            "venue",    # Predict paper venue
            "year",     # Predict publication year
            "cite"      # Predict citation relationships
        ],
        "avs": [
            "repeater"  # Predict repeat purchases
        ],
        "movielens": [
            "ratings"   # Predict user ratings on movies
        ],
        "ieeecis": [
            "fraud"     # Predict fraudulent transactions
        ],
        "diginetica": [
            "ctr",      # Click-through rate prediction
            "purchase"  # Purchase prediction
        ],
        "retailrocket": [
            "cvr"       # Conversion rate prediction
        ],
        "outbrain": [
            "ctr"       # Click-through rate prediction
        ],
        "stackexchange": [
            "upvote",   # Predict post upvotes
            "churn"     # Predict user churn
        ],
    }
}

# Default configurations
DEFAULT_CONFIG = {
    "llm_model": "sonnet4",
    "method": "autog-s", 
    "dataset": "custom",
    "task": "relation",
    "seed": 42,
    "cache_strategy": "hybrid"
}