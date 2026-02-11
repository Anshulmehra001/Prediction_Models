"""
Setup Script for Stock Prediction Models
Checks environment and installs dependencies
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_venv():
    """Check if virtual environment exists"""
    print("\n📦 Checking virtual environment...")
    venv_path = Path('.venv')
    if venv_path.exists():
        print("✅ Virtual environment found at .venv")
        return True
    print("⚠️  No .venv found. Creating one...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
        print("✅ Virtual environment created")
        return True
    except Exception as e:
        print(f"❌ Failed to create venv: {e}")
        return False

def check_env_file():
    """Check if .env file exists"""
    print("\n🔐 Checking environment configuration...")
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    
    if env_example.exists():
        print("⚠️  .env file not found. Creating from .env.example...")
        try:
            env_example.read_text()
            with open('.env', 'w') as f:
                f.write(env_example.read_text())
            print("✅ .env file created")
            print("⚠️  IMPORTANT: Edit .env and add your API keys!")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env: {e}")
            return False
    
    print("❌ No .env.example found")
    return False

def install_dependencies():
    """Install required packages"""
    print("\n📥 Installing dependencies...")
    requirements = Path('requirements.txt')
    
    if not requirements.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("Installing packages (this may take a few minutes)...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def verify_installation():
    """Verify key packages are installed"""
    print("\n🔍 Verifying installation...")
    required_packages = [
        'numpy', 'pandas', 'tensorflow', 'sklearn', 
        'xgboost', 'lightgbm', 'yfinance'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All key packages installed")
    return True

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📝 NEXT STEPS:")
    print("\n1. Edit .env file and add your API keys:")
    print("   - Get NEWS_API_KEY from https://newsapi.org/ (free)")
    print("   - Optional: ALPHA_VANTAGE_KEY from https://www.alphavantage.co/")
    print("\n2. Activate virtual environment:")
    print("   Windows: .venv\\Scripts\\activate")
    print("   macOS/Linux: source .venv/bin/activate")
    print("\n3. Run a model:")
    print('   python "SM prediction/00_ULTIMATE_Predictor.py"')
    print('   python "SM prediction/01_Phoenix_Production_Ensemble.py"')
    print("\n4. Read SETUP.md for detailed instructions")
    print("\n⚠️  REMEMBER: Paper trading only! Not financial advice!")
    print("="*60)

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 STOCK PREDICTION MODELS - SETUP")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_venv(),
        check_env_file(),
    ]
    
    if not all(checks):
        print("\n❌ Setup failed. Please fix the errors above.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages missing. Try manual installation.")
    
    print_next_steps()
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
