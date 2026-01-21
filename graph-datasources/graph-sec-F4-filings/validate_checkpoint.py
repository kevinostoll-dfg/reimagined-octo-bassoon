#!/usr/bin/env python3
"""
Validate the contents of the checkpoint PKL file.
Checks structure, data integrity, and provides statistics.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime

# Checkpoint file path
CHECKPOINT_FILE = "checkpoints_f4_processing_checkpoint.pkl"


def validate_checkpoint_structure(data: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate the structure of the checkpoint data.
    Returns (is_valid, list_of_errors)
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(data, dict):
        errors.append("❌ Checkpoint data is not a dictionary")
        return False, errors
    
    # Check for required keys
    required_keys = ['processed']
    for key in required_keys:
        if key not in data:
            errors.append(f"❌ Missing required key: '{key}'")
    
    # Validate 'processed' key
    if 'processed' in data:
        if not isinstance(data['processed'], dict):
            errors.append("❌ 'processed' key is not a dictionary")
        else:
            # Validate each entry in processed
            for file_path, entry in data['processed'].items():
                if not isinstance(entry, dict):
                    errors.append(f"❌ Entry for '{file_path}' is not a dictionary")
                    continue
                
                # Check required fields in each entry
                required_entry_fields = ['file_path', 'status', 'processed_at']
                for field in required_entry_fields:
                    if field not in entry:
                        errors.append(f"❌ Entry for '{file_path}' missing field: '{field}'")
                
                # Validate status values
                if 'status' in entry:
                    valid_statuses = ['completed', 'failed', 'error']
                    if entry['status'] not in valid_statuses:
                        errors.append(f"❌ Entry for '{file_path}' has invalid status: '{entry['status']}'")
                
                # Validate timestamps
                if 'processed_at' in entry:
                    try:
                        datetime.fromisoformat(entry['processed_at'].replace('Z', '+00:00'))
                    except (ValueError, AttributeError) as e:
                        errors.append(f"❌ Entry for '{file_path}' has invalid timestamp: {e}")
                
                # Validate numeric fields
                if 'duration' in entry and entry['duration'] is not None:
                    if not isinstance(entry['duration'], (int, float)):
                        errors.append(f"❌ Entry for '{file_path}' has invalid duration type")
                
                if 'relationships_count' in entry and entry['relationships_count'] is not None:
                    if not isinstance(entry['relationships_count'], int):
                        errors.append(f"❌ Entry for '{file_path}' has invalid relationships_count type")
    
    # Validate optional metadata fields
    if 'last_updated' in data and data['last_updated'] is not None:
        try:
            datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            errors.append(f"❌ 'last_updated' has invalid timestamp: {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def analyze_checkpoint_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze checkpoint data and return statistics.
    """
    stats = {
        'total_entries': 0,
        'status_counts': defaultdict(int),
        'total_relationships': 0,
        'total_duration': 0.0,
        'entries_with_errors': 0,
        'file_paths': [],
        'date_range': {'earliest': None, 'latest': None},
        'metadata': {}
    }
    
    processed = data.get('processed', {})
    stats['total_entries'] = len(processed)
    
    # Analyze each entry
    timestamps = []
    for file_path, entry in processed.items():
        stats['file_paths'].append(file_path)
        
        # Count by status
        status = entry.get('status', 'unknown')
        stats['status_counts'][status] += 1
        
        # Sum relationships
        if 'relationships_count' in entry and entry['relationships_count'] is not None:
            stats['total_relationships'] += entry['relationships_count']
        
        # Sum duration
        if 'duration' in entry and entry['duration'] is not None:
            stats['total_duration'] += entry['duration']
        
        # Count entries with errors
        if entry.get('error'):
            stats['entries_with_errors'] += 1
        
        # Collect timestamps for date range
        if 'processed_at' in entry and entry['processed_at']:
            try:
                ts = datetime.fromisoformat(entry['processed_at'].replace('Z', '+00:00'))
                timestamps.append(ts)
            except:
                pass
    
    # Calculate date range
    if timestamps:
        stats['date_range']['earliest'] = min(timestamps)
        stats['date_range']['latest'] = max(timestamps)
    
    # Metadata
    stats['metadata'] = {
        'last_updated': data.get('last_updated'),
        'last_saved_at': data.get('last_saved_at'),
        'last_saved_count': data.get('last_saved_count'),
        'save_in_progress': data.get('save_in_progress', False)
    }
    
    return stats


def main():
    """
    Main validation function.
    """
    print("="*80)
    print("CHECKPOINT PKL FILE VALIDATION")
    print("="*80)
    print()
    
    # Check if file exists
    checkpoint_path = Path(__file__).parent / CHECKPOINT_FILE
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print(f"   Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"📁 File: {checkpoint_path}")
    file_size = checkpoint_path.stat().st_size
    print(f"📦 Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print()
    
    # Load the pickle file
    print("📥 Loading checkpoint file...")
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print("✅ Checkpoint file loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint file: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
    
    print()
    
    # Validate structure
    print("🔍 Validating structure...")
    is_valid, errors = validate_checkpoint_structure(checkpoint_data)
    
    if is_valid:
        print("✅ Structure validation passed")
    else:
        print("❌ Structure validation failed:")
        for error in errors:
            print(f"   {error}")
        print()
        print("⚠️  Continuing with analysis despite validation errors...")
    
    print()
    
    # Analyze data
    print("📊 Analyzing checkpoint data...")
    stats = analyze_checkpoint_data(checkpoint_data)
    
    print()
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print()
    
    print(f"📋 SUMMARY:")
    print(f"   Total entries: {stats['total_entries']:,}")
    print(f"   Status breakdown:")
    for status, count in sorted(stats['status_counts'].items()):
        percentage = (count / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        print(f"      {status}: {count:,} ({percentage:.1f}%)")
    print()
    
    print(f"📈 STATISTICS:")
    if stats['total_entries'] > 0:
        print(f"   Total relationships: {stats['total_relationships']:,}")
        print(f"   Average relationships per filing: {stats['total_relationships'] / stats['total_entries']:.1f}")
        print(f"   Total processing duration: {stats['total_duration']:.1f}s")
        print(f"   Average duration per filing: {stats['total_duration'] / stats['total_entries']:.1f}s")
        print(f"   Entries with errors: {stats['entries_with_errors']:,}")
    
    if stats['date_range']['earliest'] and stats['date_range']['latest']:
        print(f"   Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    print()
    
    print(f"📝 METADATA:")
    metadata = stats['metadata']
    if metadata.get('last_updated'):
        print(f"   Last updated: {metadata['last_updated']}")
    if metadata.get('last_saved_at'):
        print(f"   Last saved at: {datetime.fromtimestamp(metadata['last_saved_at'])}")
    if metadata.get('last_saved_count') is not None:
        print(f"   Last saved count: {metadata['last_saved_count']:,}")
    if metadata.get('save_in_progress'):
        print(f"   ⚠️  Save in progress: {metadata['save_in_progress']}")
    print()
    
    # Sample entries
    print(f"🔍 SAMPLE ENTRIES (first 5):")
    processed = checkpoint_data.get('processed', {})
    for i, (file_path, entry) in enumerate(list(processed.items())[:5]):
        print(f"   {i+1}. {file_path.split('/')[-1] if '/' in file_path else file_path}")
        print(f"      Status: {entry.get('status', 'N/A')}")
        print(f"      Processed at: {entry.get('processed_at', 'N/A')}")
        if entry.get('relationships_count') is not None:
            print(f"      Relationships: {entry.get('relationships_count')}")
        if entry.get('duration') is not None:
            print(f"      Duration: {entry.get('duration'):.1f}s")
        if entry.get('error'):
            print(f"      Error: {entry.get('error')[:100]}...")
    if len(processed) > 5:
        print(f"   ... and {len(processed) - 5:,} more entries")
    print()
    
    # Final validation status
    print("="*80)
    if is_valid and stats['total_entries'] > 0:
        print("✅ VALIDATION PASSED")
    elif is_valid:
        print("⚠️  VALIDATION PASSED (but checkpoint is empty)")
    else:
        print("❌ VALIDATION FAILED")
    print("="*80)
    
    # Exit with appropriate code
    if not is_valid:
        sys.exit(1)
    elif stats['total_entries'] == 0:
        sys.exit(2)  # Empty checkpoint
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
