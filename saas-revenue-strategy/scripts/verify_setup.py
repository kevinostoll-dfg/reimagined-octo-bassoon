#!/usr/bin/env python3
"""
Verification script for SaaS Revenue Strategy RAG Agent setup.
Checks all components and dependencies.
"""

import os
import sys

# Environment variables are read from the process environment.


def check_env_variables():
    """Check if required environment variables are set."""
    print("Checking environment variables...")
    
    required_vars = [
        "NOVITA_API_KEY",
        "TAVILY_API_KEY",
    ]
    
    optional_vars = [
        "MILVUS_HOST",
        "MILVUS_PORT",
        "MILVUS_COLLECTION_NAME",
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == f"your-{var.lower().replace('_', '-')}-here":
            missing_vars.append(var)
            print(f"  ✗ {var}: Not set or using default value")
        else:
            print(f"  ✓ {var}: Set")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  ℹ {var}: Using default")
    
    if missing_vars:
        print(f"\n⚠ Warning: Required variables not set: {', '.join(missing_vars)}")
        print("Please set them in your .env or env.production file")
        return False
    
    return True


def check_milvus_connection():
    """Check if Milvus is accessible."""
    print("\nChecking Milvus connection...")
    
    try:
        from pymilvus import connections, utility
        
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        
        connections.connect(
            alias="default",
            host=host,
            port=port,
            timeout=5
        )
        
        print(f"  ✓ Connected to Milvus at {host}:{port}")
        
        # Check collection
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "saas_revenue_knowledge")
        if utility.has_collection(collection_name):
            from pymilvus import Collection
            collection = Collection(collection_name)
            print(f"  ✓ Collection '{collection_name}' exists")
            print(f"  ℹ Entities: {collection.num_entities}")
        else:
            print(f"  ⚠ Collection '{collection_name}' does not exist")
            print(f"    Run 'python scripts/setup_milvus.py' to create it")
        
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to connect to Milvus: {e}")
        print(f"    Make sure Milvus is running: docker-compose up -d")
        return False


def check_python_packages():
    """Check if required Python packages are installed."""
    print("\nChecking Python packages...")
    
    required_packages = [
        "llama_index",
        "pymilvus",
        "openai",
        "tavily",
        "dotenv",
        "yaml",
        "rich",
        "click",
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Warning: Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_docker():
    """Check if Docker is running."""
    print("\nChecking Docker...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("  ✓ Docker is running")
            
            # Check for Milvus containers
            if "milvus" in result.stdout.lower():
                print("  ✓ Milvus containers are running")
            else:
                print("  ⚠ Milvus containers not found")
                print("    Run: docker-compose up -d")
            
            return True
        else:
            print("  ✗ Docker is not accessible")
            return False
            
    except FileNotFoundError:
        print("  ✗ Docker is not installed")
        return False
    except Exception as e:
        print(f"  ✗ Error checking Docker: {e}")
        return False


def check_config_files():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")
    
    config_files = [
        "config/agents.yaml",
        "config/sources.yaml",
        ".env.example",
        "docker-compose.yml",
        "requirements.txt",
    ]
    
    all_exist = True
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}: Not found")
            all_exist = False
    
    env_found = False
    if os.path.exists(".env"):
        print("  ✓ .env")
        env_found = True

    if os.path.exists("env.production"):
        print("  ✓ env.production")
        env_found = True

    if not env_found:
        print("  ⚠ .env / env.production: Not found")
        print("    Copy .env.example to .env or set keys in env.production")
    
    return all_exist


def main():
    """Run all verification checks."""
    print("="*60)
    print("SaaS Revenue Strategy RAG Agent - Setup Verification")
    print("="*60)
    
    checks = [
        ("Configuration Files", check_config_files),
        ("Python Packages", check_python_packages),
        ("Environment Variables", check_env_variables),
        ("Docker", check_docker),
        ("Milvus Connection", check_milvus_connection),
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_passed = True
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run knowledge miner: python agents/knowledge_miner.py")
        print("  2. Query the agent: python agents/query_agent.py 'your question'")
        print("  3. Use interactive mode: python main.py --interactive")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
