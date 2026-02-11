"""
Secure Configuration Management
================================
Loads API keys and settings from environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Secure configuration class"""
    
    # API Keys (loaded from environment)
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_data.db')
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.02'))
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.03'))
    TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.06'))
    
    # Trading Mode
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    REAL_TRADING = os.getenv('REAL_TRADING', 'false').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        warnings = []
        
        if not cls.NEWS_API_KEY:
            warnings.append("⚠️  NEWS_API_KEY not set - sentiment analysis disabled")
        
        if cls.REAL_TRADING and cls.PAPER_TRADING:
            raise ValueError("❌ Cannot enable both PAPER_TRADING and REAL_TRADING")
        
        if cls.REAL_TRADING:
            print("🚨 REAL TRADING MODE ENABLED - USE WITH CAUTION!")
        else:
            print("📝 Paper trading mode (safe)")
        
        for warning in warnings:
            print(warning)
        
        return len(warnings) == 0
    
    @classmethod
    def get_news_api_key(cls):
        """Get NEWS API key with validation"""
        if not cls.NEWS_API_KEY:
            print("⚠️  No NEWS_API_KEY found. Get one free at https://newsapi.org/")
            return None
        return cls.NEWS_API_KEY


# Validate configuration on import
Config.validate()
