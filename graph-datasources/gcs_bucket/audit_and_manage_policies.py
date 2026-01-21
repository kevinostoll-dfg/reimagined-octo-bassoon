#!/usr/bin/env python3
"""
Audit and manage GCS bucket lifecycle and retention policies.

This script audits the current lifecycle and retention policies for the
blacksmith-sec-filings bucket and can create/update them as needed.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from google.cloud import storage
from google.api_core import exceptions


# Configuration
BUCKET_NAME = "blacksmith-sec-filings"
DEFAULT_GCP_PROJECT = "blacksmith-443107"
BACKUP_DIR = "./backups"

# Default policy configurations
# Set to None to disable retention policy by default (allows object updates)
DEFAULT_RETENTION_DAYS = None

# Default lifecycle policy: Move to Nearline at 35 days, Coldline at 45 days, Delete at 60 days
DEFAULT_LIFECYCLE_POLICY = {
    "rule": [
        {
            "action": {
                "type": "SetStorageClass",
                "storageClass": "NEARLINE"
            },
            "condition": {
                "age": 35
            }
        },
        {
            "action": {
                "type": "SetStorageClass",
                "storageClass": "COLDLINE"
            },
            "condition": {
                "age": 45
            }
        },
        {
            "action": {
                "type": "Delete"
            },
            "condition": {
                "age": 60
            }
        }
    ]
}


def get_bucket_info(bucket: storage.Bucket) -> Dict[str, Any]:
    """
    Get comprehensive information about the bucket.
    
    Args:
        bucket: GCS bucket object
        
    Returns:
        Dictionary with bucket information
    """
    bucket.reload()
    
    # Safely get IAM configuration
    uniform_bucket_level_access = None
    public_access_prevention = None
    
    if bucket.iam_configuration:
        try:
            # Try to access uniform_bucket_level_access
            if hasattr(bucket.iam_configuration, 'uniform_bucket_level_access'):
                ubla = bucket.iam_configuration.uniform_bucket_level_access
                if hasattr(ubla, 'enabled'):
                    uniform_bucket_level_access = ubla.enabled
        except (AttributeError, TypeError):
            pass
        
        try:
            # Try to access public_access_prevention
            if hasattr(bucket.iam_configuration, 'public_access_prevention'):
                public_access_prevention = bucket.iam_configuration.public_access_prevention
        except (AttributeError, TypeError):
            pass
    
    info = {
        "name": bucket.name,
        "location": bucket.location,
        "storage_class": bucket.storage_class,
        "created": bucket.time_created.isoformat() if bucket.time_created else None,
        "updated": bucket.updated.isoformat() if bucket.updated else None,
        "retention_policy": None,
        "lifecycle_rules": None,
        "uniform_bucket_level_access": uniform_bucket_level_access,
        "public_access_prevention": public_access_prevention,
    }
    
    # Get retention policy
    # Retention policy is accessed through _properties
    retention_policy = getattr(bucket, '_properties', {}).get('retentionPolicy')
    if retention_policy:
        retention_seconds = int(retention_policy.get('retentionPeriod', 0))
        retention_days = retention_seconds / (24 * 60 * 60)
        info["retention_policy"] = {
            "retention_period_seconds": retention_seconds,
            "retention_period_days": retention_days,
            "effective_time": retention_policy.get('effectiveTime'),
            "is_locked": retention_policy.get('isLocked', False)
        }
    
    # Get lifecycle rules
    # Lifecycle rules are stored as a list of dictionaries
    try:
        lifecycle_rules = getattr(bucket, 'lifecycle_rules', None)
        if lifecycle_rules:
            info["lifecycle_rules"] = []
            for rule in lifecycle_rules:
                # Rule should be a dictionary, but handle both cases
                if isinstance(rule, dict):
                    rule_dict = {
                        "action": rule.get("action", {}),
                        "condition": rule.get("condition", {})
                    }
                else:
                    # Convert rule object to dict if needed
                    rule_dict = {
                        "action": dict(rule.action) if hasattr(rule, 'action') else {},
                        "condition": dict(rule.condition) if hasattr(rule, 'condition') else {}
                    }
                
                info["lifecycle_rules"].append(rule_dict)
    except (AttributeError, TypeError) as e:
        # If lifecycle_rules can't be accessed, leave it as None
        pass
    
    return info


def audit_bucket_policies(bucket_name: str, gcp_project: Optional[str] = None) -> Dict[str, Any]:
    """
    Audit the current lifecycle and retention policies of a GCS bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        gcp_project: Optional GCP project ID
        
    Returns:
        Dictionary with audit results
    """
    print(f"\n{'='*80}")
    print(f"AUDITING BUCKET: gs://{bucket_name}")
    print(f"{'='*80}\n")
    
    try:
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket gs://{bucket_name} does not exist")
        
        info = get_bucket_info(bucket)
        
        # Print audit results
        print(f"📦 Bucket Information:")
        print(f"   Name: {info['name']}")
        print(f"   Location: {info['location']}")
        print(f"   Storage Class: {info['storage_class']}")
        print(f"   Created: {info['created']}")
        print(f"   Updated: {info['updated']}")
        print(f"   Uniform Bucket-Level Access: {info['uniform_bucket_level_access']}")
        print(f"   Public Access Prevention: {info['public_access_prevention']}")
        print()
        
        # Retention Policy
        print(f"🔒 Retention Policy:")
        if info['retention_policy']:
            retention = info['retention_policy']
            print(f"   ✅ Retention Period: {retention['retention_period_days']:.1f} days ({retention['retention_period_seconds']} seconds)")
            if retention['effective_time']:
                print(f"   Effective Time: {retention['effective_time']}")
            if retention.get('is_locked'):
                print(f"   ⚠️  WARNING: Retention policy is LOCKED")
        else:
            print(f"   ❌ No retention policy configured")
        print()
        
        # Lifecycle Rules
        print(f"🔄 Lifecycle Rules:")
        if info['lifecycle_rules']:
            print(f"   ✅ {len(info['lifecycle_rules'])} rule(s) configured:")
            for i, rule in enumerate(info['lifecycle_rules'], 1):
                action_type = rule['action']['type']
                storage_class = rule['action'].get('storageClass', 'N/A')
                age = rule['condition'].get('age', 'N/A')
                
                if action_type == 'SetStorageClass':
                    print(f"   {i}. Move to {storage_class} storage after {age} days")
                elif action_type == 'Delete':
                    print(f"   {i}. Delete objects after {age} days")
                else:
                    print(f"   {i}. {action_type} after {age} days")
        else:
            print(f"   ❌ No lifecycle rules configured")
        print()
        
        return {
            "success": True,
            "bucket_info": info
        }
        
    except exceptions.NotFound:
        return {
            "success": False,
            "error": f"Bucket gs://{bucket_name} not found"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def backup_current_policies(bucket_name: str, gcp_project: Optional[str] = None) -> Optional[str]:
    """
    Backup current bucket policies to a JSON file.
    
    Args:
        bucket_name: Name of the GCS bucket
        gcp_project: Optional GCP project ID
        
    Returns:
        Path to backup file, or None if backup failed
    """
    try:
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket gs://{bucket_name} does not exist")
        
        info = get_bucket_info(bucket)
        
        # Create backup directory
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create backup file
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        backup_file = Path(BACKUP_DIR) / f"{timestamp}-{bucket_name}-policies.json"
        
        backup_data = {
            "timestamp": timestamp,
            "bucket_name": bucket_name,
            "gcp_project": gcp_project or "default",
            "policies": {
                "retention_policy": info.get("retention_policy"),
                "lifecycle_rules": info.get("lifecycle_rules")
            },
            "bucket_info": info
        }
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print(f"✅ Backup created: {backup_file}")
        return str(backup_file)
        
    except Exception as e:
        print(f"❌ Failed to create backup: {e}", file=sys.stderr)
        return None


def remove_retention_policy(
    bucket_name: str,
    gcp_project: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """
    Remove retention policy from the bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        gcp_project: Optional GCP project ID
        dry_run: If True, only show what would be done
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"{'[DRY RUN] ' if dry_run else ''}REMOVING RETENTION POLICY")
    print(f"{'='*80}")
    print(f"Bucket: gs://{bucket_name}")
    print()
    
    if dry_run:
        print("🔍 DRY RUN: Would remove retention policy (not actually removed)")
        return True
    
    try:
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket gs://{bucket_name} does not exist")
        
        # Check current retention policy
        bucket.reload()
        retention_policy = getattr(bucket, '_properties', {}).get('retentionPolicy')
        if not retention_policy:
            print(f"✅ No retention policy configured. Nothing to remove.")
            return True
        
        if retention_policy.get('isLocked', False):
            print(f"⚠️  WARNING: Retention policy is LOCKED and cannot be removed!")
            return False
        
        # Remove retention policy by setting it to None
        bucket.retention_period = None
        bucket.patch()
        
        print(f"✅ Retention policy removed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to remove retention policy: {e}", file=sys.stderr)
        return False


def apply_retention_policy(
    bucket_name: str,
    retention_days: Optional[int],
    gcp_project: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """
    Apply retention policy to the bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        retention_days: Retention period in days (None to skip)
        gcp_project: Optional GCP project ID
        dry_run: If True, only show what would be done
        
    Returns:
        True if successful, False otherwise
    """
    # Skip if retention_days is None
    if retention_days is None:
        return True
    
    retention_seconds = retention_days * 24 * 60 * 60
    
    print(f"\n{'='*80}")
    print(f"{'[DRY RUN] ' if dry_run else ''}APPLYING RETENTION POLICY")
    print(f"{'='*80}")
    print(f"Bucket: gs://{bucket_name}")
    print(f"Retention Period: {retention_days} days ({retention_seconds} seconds)")
    print()
    
    if dry_run:
        print("🔍 DRY RUN: Would apply retention policy (not actually applied)")
        return True
    
    try:
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket gs://{bucket_name} does not exist")
        
        # Check current retention policy
        bucket.reload()
        retention_policy = getattr(bucket, '_properties', {}).get('retentionPolicy')
        if retention_policy:
            current_retention = int(retention_policy.get('retentionPeriod', 0))
            current_days = current_retention / (24 * 60 * 60)
            print(f"📋 Current retention: {current_days:.1f} days")
            
            if current_retention == retention_seconds:
                print(f"✅ Retention policy already set to {retention_days} days. No change needed.")
                return True
            
            if retention_policy.get('isLocked', False):
                print(f"⚠️  WARNING: Retention policy is LOCKED and cannot be changed!")
                return False
        
        # Apply retention policy
        bucket.retention_period = retention_seconds
        bucket.patch()
        
        print(f"✅ Retention policy applied successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply retention policy: {e}", file=sys.stderr)
        return False


def apply_lifecycle_policy(
    bucket_name: str,
    lifecycle_policy: Dict[str, Any],
    gcp_project: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """
    Apply lifecycle policy to the bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        lifecycle_policy: Lifecycle policy dictionary
        gcp_project: Optional GCP project ID
        dry_run: If True, only show what would be done
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"{'[DRY RUN] ' if dry_run else ''}APPLYING LIFECYCLE POLICY")
    print(f"{'='*80}")
    print(f"Bucket: gs://{bucket_name}")
    print(f"Rules: {len(lifecycle_policy.get('rule', []))}")
    print()
    
    # Print lifecycle rules
    for i, rule in enumerate(lifecycle_policy.get('rule', []), 1):
        action_type = rule.get('action', {}).get('type')
        storage_class = rule.get('action', {}).get('storageClass', 'N/A')
        age = rule.get('condition', {}).get('age', 'N/A')
        
        if action_type == 'SetStorageClass':
            print(f"   {i}. Move to {storage_class} storage after {age} days")
        elif action_type == 'Delete':
            print(f"   {i}. Delete objects after {age} days")
        else:
            print(f"   {i}. {action_type} after {age} days")
    print()
    
    if dry_run:
        print("🔍 DRY RUN: Would apply lifecycle policy (not actually applied)")
        return True
    
    try:
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket gs://{bucket_name} does not exist")
        
        # Convert policy to GCS format
        # The Google Cloud Storage library expects lifecycle rules as a list of dictionaries
        lifecycle_rules = []
        for rule in lifecycle_policy.get('rule', []):
            gcs_rule = {}
            
            # Action
            action = rule.get('action', {})
            if action.get('type') == 'SetStorageClass':
                gcs_rule['action'] = {
                    'type': 'SetStorageClass',
                    'storageClass': action.get('storageClass')
                }
            elif action.get('type') == 'Delete':
                gcs_rule['action'] = {'type': 'Delete'}
            else:
                # Handle other action types
                gcs_rule['action'] = action
            
            # Condition
            condition = rule.get('condition', {})
            gcs_rule['condition'] = {}
            if 'age' in condition:
                gcs_rule['condition']['age'] = condition['age']
            if 'matchesStorageClass' in condition:
                gcs_rule['condition']['matchesStorageClass'] = condition['matchesStorageClass']
            if 'matchesPrefix' in condition:
                gcs_rule['condition']['matchesPrefix'] = condition['matchesPrefix']
            if 'matchesSuffix' in condition:
                gcs_rule['condition']['matchesSuffix'] = condition['matchesSuffix']
            if 'createdBefore' in condition:
                gcs_rule['condition']['createdBefore'] = condition['createdBefore']
            if 'numNewerVersions' in condition:
                gcs_rule['condition']['numNewerVersions'] = condition['numNewerVersions']
            
            lifecycle_rules.append(gcs_rule)
        
        # Apply lifecycle policy
        bucket.lifecycle_rules = lifecycle_rules
        bucket.patch()
        
        print(f"✅ Lifecycle policy applied successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply lifecycle policy: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def load_lifecycle_policy_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load lifecycle policy from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Lifecycle policy dictionary, or None if failed
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both formats: direct rule array or wrapped in lifecycle object
        if 'lifecycle' in data:
            return data['lifecycle']
        elif 'rule' in data:
            return data
        else:
            raise ValueError("Invalid lifecycle policy format")
            
    except Exception as e:
        print(f"❌ Failed to load lifecycle policy from {file_path}: {e}", file=sys.stderr)
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Audit and manage GCS bucket lifecycle and retention policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit current policies:
  python audit_and_manage_policies.py --audit

  # Apply lifecycle policy only (no retention):
  python audit_and_manage_policies.py --apply

  # Apply lifecycle policy with retention (dry run):
  python audit_and_manage_policies.py --apply --retention-days 30 --dry-run

  # Apply custom retention period:
  python audit_and_manage_policies.py --apply --retention-days 60

  # Remove existing retention policy:
  python audit_and_manage_policies.py --remove-retention

  # Apply custom lifecycle policy from file:
  python audit_and_manage_policies.py --apply --lifecycle-file custom-policy.json

  # Backup current policies:
  python audit_and_manage_policies.py --backup
        """
    )
    
    parser.add_argument(
        '--bucket', '-b',
        type=str,
        default=BUCKET_NAME,
        help=f'GCS bucket name (default: {BUCKET_NAME})'
    )
    parser.add_argument(
        '--gcp-project', '-p',
        type=str,
        default=None,
        help=f'GCP project ID (default: uses default credentials)'
    )
    parser.add_argument(
        '--audit', '-a',
        action='store_true',
        help='Audit current bucket policies'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply lifecycle and retention policies'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup current policies to a JSON file'
    )
    parser.add_argument(
        '--retention-days', '-r',
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f'Retention period in days (default: None - no retention policy applied)'
    )
    parser.add_argument(
        '--remove-retention',
        action='store_true',
        help='Remove existing retention policy from the bucket'
    )
    parser.add_argument(
        '--lifecycle-file', '-l',
        type=str,
        default=None,
        help='Path to JSON file with lifecycle policy (default: uses built-in policy)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually applying changes'
    )
    
    args = parser.parse_args()
    
    # If no action specified, default to audit
    if not (args.audit or args.apply or args.backup or args.remove_retention):
        args.audit = True
    
    try:
        # Audit
        if args.audit:
            result = audit_bucket_policies(args.bucket, args.gcp_project)
            if not result.get('success'):
                print(f"❌ Audit failed: {result.get('error')}", file=sys.stderr)
                sys.exit(1)
        
        # Backup
        if args.backup:
            backup_file = backup_current_policies(args.bucket, args.gcp_project)
            if not backup_file:
                sys.exit(1)
        
        # Remove retention policy
        if args.remove_retention:
            retention_removal_success = remove_retention_policy(
                args.bucket,
                args.gcp_project,
                args.dry_run
            )
            if not retention_removal_success:
                sys.exit(1)
        
        # Apply policies
        if args.apply:
            # Load lifecycle policy
            if args.lifecycle_file:
                lifecycle_policy = load_lifecycle_policy_from_file(args.lifecycle_file)
                if not lifecycle_policy:
                    sys.exit(1)
            else:
                lifecycle_policy = DEFAULT_LIFECYCLE_POLICY
            
            # Apply retention policy (only if retention_days is specified)
            retention_success = True
            if args.retention_days is not None:
                retention_success = apply_retention_policy(
                    args.bucket,
                    args.retention_days,
                    args.gcp_project,
                    args.dry_run
                )
            
            # Apply lifecycle policy
            lifecycle_success = apply_lifecycle_policy(
                args.bucket,
                lifecycle_policy,
                args.gcp_project,
                args.dry_run
            )
            
            if not (retention_success and lifecycle_success):
                sys.exit(1)
            
            if not args.dry_run:
                print(f"\n{'='*80}")
                print(f"✅ POLICIES APPLIED SUCCESSFULLY")
                print(f"{'='*80}")
                print(f"📁 Bucket: gs://{args.bucket}")
                if args.retention_days is not None:
                    print(f"🔒 Retention: {args.retention_days} days")
                else:
                    print(f"🔒 Retention: None (no retention policy)")
                print(f"🔄 Lifecycle: {len(lifecycle_policy.get('rule', []))} rules")
                print(f"{'='*80}\n")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

