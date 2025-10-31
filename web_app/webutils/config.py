
"""
Configuration settings for AutoG2 Web App
Based on the actual AutoG2 backend pipeline
"""

# LLM Model Configurations (matching main.autog2 exactly)
LLM_MODELS = {
    "sonnet3": {
        "name": "Claude 3 Sonnet",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "description": "Reliable model for table analysis and relationship extraction"
    },
    "llama3": {
        "name": "Llama 3 70B",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "description": "Open-source model with strong performance"
    },
    "mistralarge": {
        "name": "Mistral Large",
        "model_id": "mistral.mistral-large-2402-v1:0",
        "description": "High-performance multilingual model"
    },
    "sonnet37": {
        "name": "Claude 3.7 Sonnet",
        "model_id": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "description": "Enhanced Claude 3 Sonnet with improved capabilities"
    },
    "sonnet4": {
        "name": "Claude Sonnet 4",
        "model_id": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "description": "Latest Claude Sonnet 4 model for advanced reasoning"
    },
    "opus3": {
        "name": "Claude 3 Opus",
        "model_id": "anthropic.claude-3-opus-20240229-v1:0",
        "description": "Most capable Claude 3 model for complex tasks"
    },
    "opus4": {
        "name": "Claude Opus 4",
        "model_id": "anthropic.claude-opus-4-20250514-v1:0",
        "description": "Latest Claude Opus 4 model with maximum capabilities"
    },
    "haiku3": {
        "name": "Claude 3 Haiku",
        "model_id": "anthropic.claude-3-haiku-20240229-v1:0",
        "description": "Fast and efficient model for simpler analysis tasks"
    },
    "sonnet45": {
        "name": "Claude Sonnet 4.5",
        "model_id": "arn:aws:bedrock:us-west-2:911734752298:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "description": "Latest Claude Sonnet 4.5 with enhanced performance"
    }
}

# AutoG Configuration (matching actual task definitions)
AUTOG_CONFIG = {
    "methods": [
        "autog-s",  # Single-round AutoG
        "autog-m",  # Multi-round AutoG
        "baseline"
    ],
    "datasets": {
        "mag": [
            "venue",    # Predict paper venue
            "year",     # Predict publication year
            "cite"      # Predict citation relationships
        ],
        "movielens": [
            "ratings"   # Predict user ratings on movies
        ],
        "avs": [
            "repeater"  # Predict repeat purchases
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
        "adsp": [
            "kg"        # Knowledge graph construction
        ],
        "custom": [
            "relation", # Find primary/foreign keys
            "kg",       # Knowledge graph construction
            "kg2"       # Enhanced knowledge graph construction
        ],
        "custom_mag": [
            "venue"     # Custom MAG venue prediction
        ]
    }
}

# Default configurations
DEFAULT_CONFIG = {
    "llm_model": "sonnet3",
    "method": "autog-s", 
    "task": "custom:kg",
    "seed": 0,
    "cache_strategy": "hybrid"
}