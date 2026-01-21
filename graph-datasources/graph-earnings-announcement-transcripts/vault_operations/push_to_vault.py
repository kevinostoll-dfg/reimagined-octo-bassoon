#!/usr/bin/env python3
"""
Script to push environment variables to HashiCorp Vault for graph-earnings-announcement-transcripts.

This script:
1. Parses Python scripts to extract all os.getenv() calls
2. Reads from .env file if it exists
3. Creates a snapshot of all environment variables
4. Pushes them to Vault at the configured path

Usage:
    export VAULT_TOKEN="your-vault-token"
    python push_to_vault.py [--dry-run]

Requirements:
- hvac library: pip install hvac
- VAULT_TOKEN environment variable set
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Set, Optional, Tuple
from dotenv import load_dotenv

try:
    import hvac
except ImportError:
    print("❌ Error: hvac library not installed.", file=sys.stderr)
    print("   Install it with: pip install hvac", file=sys.stderr)
    sys.exit(1)

# Vault configuration
VAULT_MOUNT_POINT = "blacksmith-project-secrets"  # The KV v2 mount point
VAULT_SECRET_PATH = "graph-earnings-announcement-transcripts"  # The path within the mount point
VAULT_URL = os.getenv("VAULT_ADDR", "https://vault.zagreus.deerfieldgreen.com")

# Script directory (parent of vault_operations)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Runtime variables to exclude (these are set at runtime, not stored as secrets)
RUNTIME_VARS = {"SYMBOL", "YEAR", "QUARTER", "PYTHON_EXECUTABLE"}


def get_vault_client() -> hvac.Client:
    """Initialize and authenticate Vault client."""
    vault_token = os.getenv("VAULT_TOKEN")
    
    if not vault_token:
        print("❌ Error: VAULT_TOKEN environment variable is not set.", file=sys.stderr)
        print("   Set it with: export VAULT_TOKEN='your-token-here'", file=sys.stderr)
        sys.exit(1)
    
    client = hvac.Client(url=VAULT_URL, token=vault_token)
    
    # Verify authentication
    if not client.is_authenticated():
        print("❌ Error: Failed to authenticate with Vault.", file=sys.stderr)
        print(f"   Vault URL: {VAULT_URL}", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Authenticated with Vault at {VAULT_URL}")
    return client


def parse_scripts_for_env_vars() -> Dict[str, Optional[str]]:
    """
    Parse Python scripts to extract all os.getenv() calls and their default values.
    Returns a dictionary mapping variable names to their default values (None if no default).
    """
    env_vars = {}  # var_name -> default_value
    
    # Scripts to parse
    scripts_to_parse = [
        PROJECT_DIR / "v1.0-graph-ea-scripts.py",
    ]
    
    # Also check load_data_gcs directory
    load_data_dir = PROJECT_DIR / "load_data_gcs"
    if load_data_dir.exists():
        for script_file in load_data_dir.glob("*.py"):
            scripts_to_parse.append(script_file)
    
    print("\n📖 Parsing scripts for environment variables...")
    print("-" * 80)
    
    # Pattern to match os.getenv("VAR_NAME", "default") or os.getenv('VAR_NAME', 'default')
    # Also matches os.getenv("VAR_NAME") without default
    # Matches: os.getenv("VAR", "default") or os.getenv("VAR", 'default') or os.getenv("VAR")
    pattern_with_default = r'os\.getenv\(["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']'
    pattern_without_default = r'os\.getenv\(["\']([^"\']+)["\']\s*\)'
    
    for script_path in scripts_to_parse:
        if not script_path.exists():
            continue
        
        print(f"  Parsing: {script_path.name}")
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find matches with defaults first
            matches_with_default = re.findall(pattern_with_default, content)
            for var_name, default_value in matches_with_default:
                var_name = var_name.strip()
                default_value = default_value.strip()
                if var_name and var_name not in RUNTIME_VARS:
                    # Only update if not already found (first occurrence wins)
                    if var_name not in env_vars:
                        env_vars[var_name] = default_value
                        print(f"    Found: {var_name} (default: {default_value})")
            
            # Find matches without defaults
            matches_without_default = re.findall(pattern_without_default, content)
            for var_name in matches_without_default:
                var_name = var_name.strip()
                if var_name and var_name not in RUNTIME_VARS:
                    # Only add if not already found
                    if var_name not in env_vars:
                        env_vars[var_name] = None
                        print(f"    Found: {var_name} (no default)")
        
        except Exception as e:
            print(f"    ⚠️  Error parsing {script_path.name}: {e}")
    
    return env_vars


def load_env_file() -> Dict[str, str]:
    """
    Load environment variables from .env file if it exists.
    Returns a dictionary of variable names to values.
    Note: This is used only for getting values, not for discovering new variables.
    Only variables found in scripts will be included in the final secrets.
    """
    env_vars = {}
    
    env_file = PROJECT_DIR / ".env"
    
    if env_file.exists():
        print(f"\n📄 Loading .env file: {env_file}")
        print("-" * 80)
        
        # Use python-dotenv to load the file
        load_dotenv(dotenv_path=env_file, override=False)
        
        # Read the file directly to get all variables
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        if key and key not in RUNTIME_VARS:
                            env_vars[key] = value
                            # Mask sensitive values in display
                            display_value = "***" if any(kw in key.lower() for kw in ["password", "secret", "token", "key"]) else value
                            print(f"  {key:45} = {display_value}")
        except Exception as e:
            print(f"  ⚠️  Error reading .env file: {e}")
    else:
        print(f"\n📄 No .env file found at: {env_file}")
    
    return env_vars


def collect_secrets() -> Tuple[Dict[str, Any], Dict[str, Optional[str]]]:
    """
    Collect secrets by:
    1. Parsing scripts to find all environment variables and their defaults (ONLY these are included)
    2. Loading values from .env file or current environment
    3. Using script defaults as fallback
    4. Creating a snapshot
    
    Returns a dictionary of all secrets to push.
    Only includes variables that are actually used in the scripts.
    """
    secrets = {}
    
    # Step 1: Parse scripts to find all environment variables and their defaults
    # Returns dict: var_name -> default_value (or None if no default)
    script_env_vars_with_defaults = parse_scripts_for_env_vars()
    
    # Step 2: Load from .env file (for values only, not for discovering new variables)
    env_file_vars = load_env_file()
    
    # Step 3: Only use variables found in scripts
    all_var_names = set(script_env_vars_with_defaults.keys())
    
    print(f"\n📋 Collecting environment variables ({len(all_var_names)} total):")
    print("-" * 80)
    
    # Step 4: Collect values (priority: current env > .env file > script default > empty string)
    for var_name in sorted(all_var_names):
        # Priority: current environment > .env file > script default > empty string
        value = os.getenv(var_name)
        used_default = False
        
        if value is None and var_name in env_file_vars:
            value = env_file_vars[var_name]
        
        # Use script default if available
        if value is None and var_name in script_env_vars_with_defaults:
            script_default = script_env_vars_with_defaults[var_name]
            if script_default is not None:
                value = script_default
                used_default = True
        
        # If still None, use empty string (required but no default)
        if value is None:
            value = ""
        
        secrets[var_name] = str(value)
        
        # Display (mask sensitive values)
        has_default = script_env_vars_with_defaults.get(var_name) is not None
        is_empty = not value
        
        if any(keyword in var_name.lower() for keyword in ["password", "secret", "token", "key"]):
            display_value = "***" if value else "(empty - REQUIRED)"
        else:
            if used_default:
                display_value = f"{value} (from script default)"
            elif value:
                display_value = str(value)
            else:
                display_value = "(empty - REQUIRED)"
        
        # Only mark as REQUIRED if it has no default and is empty
        status = "[REQUIRED]" if (is_empty and not has_default) else ""
        print(f"  {var_name:45} = {display_value:30} {status}")
    
    return secrets, script_env_vars_with_defaults


def push_to_vault(secrets: Dict[str, Any], dry_run: bool = False) -> bool:
    """Push secrets to Vault."""
    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Would push the following to Vault:")
        print("=" * 80)
        print(f"Mount Point: {VAULT_MOUNT_POINT}")
        print(f"Secret Path: {VAULT_SECRET_PATH}")
        print("\nSecrets:")
        for key, value in sorted(secrets.items()):
            if any(keyword in key.lower() for keyword in ["password", "secret", "token", "key"]):
                print(f"  {key} = ***")
            else:
                print(f"  {key} = {value}")
        print()
        return True
    
    try:
        client = get_vault_client()
        
        print(f"\n📤 Pushing secrets to Vault:")
        print(f"  Mount Point: {VAULT_MOUNT_POINT}")
        print(f"  Secret Path: {VAULT_SECRET_PATH}")
        print("-" * 80)
        
        # Push to Vault (KV v2)
        # Note: mount_point is the KV v2 mount, path is the secret path within that mount
        response = client.secrets.kv.v2.create_or_update_secret(
            mount_point=VAULT_MOUNT_POINT,
            path=VAULT_SECRET_PATH,
            secret=secrets
        )
        
        if response:
            print("✅ Successfully pushed secrets to Vault!")
            
            # Verify by reading back
            print("\n🔍 Verifying secrets were stored...")
            read_response = client.secrets.kv.v2.read_secret_version(
                mount_point=VAULT_MOUNT_POINT,
                path=VAULT_SECRET_PATH,
                raise_on_deleted_version=True
            )
            
            if read_response and 'data' in read_response:
                stored_secrets = read_response['data']['data']
                print(f"✅ Verified: {len(stored_secrets)} secrets stored in Vault")
                return True
            else:
                print("⚠️  Warning: Could not verify secrets were stored")
                return False
        else:
            print("❌ Error: Failed to push secrets to Vault")
            return False
            
    except Exception as e:
        print(f"❌ Error pushing to Vault: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Push graph-earnings-announcement-transcripts environment variables to Vault"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pushed without actually pushing"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Graph Earnings Announcement Transcripts - Push to Vault")
    print("=" * 80)
    print(f"Vault URL: {VAULT_URL}")
    print(f"Mount Point: {VAULT_MOUNT_POINT}")
    print(f"Secret Path: {VAULT_SECRET_PATH}")
    print(f"Project Directory: {PROJECT_DIR}")
    print()
    
    # Collect secrets
    secrets, script_env_vars_with_defaults = collect_secrets()
    
    if not secrets:
        print("❌ Error: No secrets to push", file=sys.stderr)
        sys.exit(1)
    
    # Check for required variables that are empty (no default value)
    missing_required = [
        k for k, v in secrets.items() 
        if not v and script_env_vars_with_defaults.get(k) is None
    ]
    if missing_required:
        print(f"\n⚠️  Warning: {len(missing_required)} required variables are empty (no default value):")
        for var in missing_required:
            print(f"    - {var}")
        print("\nThese will be stored as empty strings. You may need to set them manually in Vault.")
    
    # Push to Vault
    success = push_to_vault(secrets, dry_run=args.dry_run)
    
    if success:
        print("\n" + "=" * 80)
        if args.dry_run:
            print("✅ Dry run completed successfully")
            print("   Run without --dry-run to actually push to Vault")
        else:
            print("✅ Secrets successfully pushed to Vault!")
            print(f"\nTo use these secrets in production, ensure your application")
            print(f"can read from Vault at mount: {VAULT_MOUNT_POINT}, path: {VAULT_SECRET_PATH}")
            print(f"\nVault UI: https://vault.zagreus.deerfieldgreen.com/ui/vault/secrets/{VAULT_MOUNT_POINT}/show/{VAULT_SECRET_PATH}")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ Failed to push secrets to Vault")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()










