#!/usr/bin/env python3
"""
Setup script for Railway deployment
Run this before pushing to GitHub and deploying on Railway
"""

import os
import sys
import subprocess

def run_command(command):
    """Run a command and return True if successful"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Preparing Stel Docs AI for Railway deployment...")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('manage.py'):
        print("❌ Error: Please run this script from the Django project root directory")
        sys.exit(1)
    
    # Run Django checks
    print("1. Running Django system checks...")
    if not run_command("python manage.py check"):
        print("❌ Django checks failed. Please fix errors before deploying.")
        return False
    
    # Collect static files
    print("\n2. Collecting static files...")
    if not run_command("python manage.py collectstatic --noinput"):
        print("❌ Static file collection failed.")
        return False
    
    # Check if database exists
    if os.path.exists('db.sqlite3'):
        print("✅ 3. Database file found: db.sqlite3")
    else:
        print("⚠️  3. No database found. Running migrations...")
        if not run_command("python manage.py migrate"):
            print("❌ Migrations failed.")
            return False
    
    # Check ChromaDB directory
    if os.path.exists('chroma_db'):
        print("✅ 4. ChromaDB directory found: chroma_db/")
    else:
        print("✅ 4. ChromaDB directory will be created automatically")
    
    # Check environment variables
    print("\n5. Checking environment variables...")
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"✅ Environment file found: {env_file}")
    else:
        print("⚠️  Creating .env file template...")
        with open(env_file, 'w') as f:
            f.write("""# Environment variables for Railway deployment
# Railway will auto-generate SECRET_KEY, but you can set it manually if needed
SECRET_KEY=change-this-to-a-random-secret-key-or-let-railway-generate
DEBUG=False
GOOGLE_API_KEY=your-google-gemini-api-key-here
PYTHONUNBUFFERED=1
""")
        print(f"✅ Created {env_file} - Please update with your actual values!")
    
    print("\n🎉 Deployment preparation complete!")
    print("\n📋 Next steps:")
    print("1. Update your .env file with your actual Google API key")
    print("2. Push your code to GitHub")
    print("3. Connect GitHub to Railway and deploy")
    print("4. Follow the RAILWAY_DEPLOY.md guide")
    print("5. Your SQLite database and ChromaDB embeddings will persist automatically!")
    print("\n🚀 Railway is perfect for AI/ML apps with 8GB RAM and persistent storage!")
    
    return True

if __name__ == "__main__":
    main() 