"""
Quick Setup Verification Script
Checks if everything is configured correctly
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has keys"""
    print("🔐 Checking .env configuration...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Run: python setup.py")
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        news_key = os.getenv('NEWS_API_KEY', '')
        alpha_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        
        if news_key:
            print(f"✅ NEWS_API_KEY: {news_key[:10]}...{news_key[-4:]}")
        else:
            print("⚠️  NEWS_API_KEY not set (sentiment analysis disabled)")
        
        if alpha_key:
            print(f"✅ ALPHA_VANTAGE_KEY: {alpha_key[:10]}...{alpha_key[-4:]}")
        else:
            print("⚠️  ALPHA_VANTAGE_KEY not set (optional)")
        
        paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        real_trading = os.getenv('REAL_TRADING', 'false').lower() == 'true'
        
        if paper_trading:
            print("✅ Paper trading mode: ENABLED (safe)")
        if real_trading:
            print("🚨 Real trading mode: ENABLED (DANGEROUS!)")
        
        return True
        
    except ImportError:
        print("❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False

def check_packages():
    """Check if key packages are installed"""
    print("\n📦 Checking packages...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'yfinance': 'yfinance',
        'transformers': 'Transformers',
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_models():
    """Check if model files exist"""
    print("\n📊 Checking model files...")
    
    models = [
        "SM prediction/00_ULTIMATE_Predictor.py",
        "SM prediction/01_Phoenix_Production_Ensemble.py",
        "SM prediction/06_Technical_Indicators_Signals.py",
    ]
    
    for model in models:
        if Path(model).exists():
            print(f"✅ {Path(model).name}")
        else:
            print(f"❌ {Path(model).name} - NOT FOUND")
            return False
    
    return True

def test_data_fetch():
    """Test if we can fetch stock data"""
    print("\n🌐 Testing data connection...")
    
    try:
        import yfinance as yf
        
        # Try to fetch a small amount of data
        ticker = yf.Ticker("RELIANCE.NS")
        hist = ticker.history(period="5d")
        
        if not hist.empty:
            print(f"✅ Successfully fetched data for RELIANCE.NS")
            print(f"   Latest close: ₹{hist['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("⚠️  No data returned (market might be closed)")
            return True  # Not a critical error
            
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("🔍 SETUP VERIFICATION")
    print("="*60)
    print()
    
    checks = {
        'Environment': check_env_file(),
        'Packages': check_packages(),
        'Models': check_models(),
        'Data Connection': test_data_fetch(),
    }
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    if all(checks.values()):
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ You're ready to run the models!")
        print("\n📝 Quick start:")
        print('   python "SM prediction/00_ULTIMATE_Predictor.py"')
        print('   python "SM prediction/01_Phoenix_Production_Ensemble.py"')
        print("\n⚠️  Remember: Paper trading only!")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("   Run: python setup.py")
    
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
