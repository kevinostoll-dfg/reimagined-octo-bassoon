#!/usr/bin/env python3
"""
Detailed validation of checkpoint PKL file - checks for duplicates, inconsistencies, etc.
"""

import pickle
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any

CHECKPOINT_FILE = "checkpoints_f4_processing_checkpoint.pkl"


def detailed_analysis(checkpoint_data: Dict[str, Any]):
    """
    Perform detailed analysis of checkpoint data.
    """
    processed = checkpoint_data.get('processed', {})
    
    print("="*80)
    print("DETAILED CHECKPOINT ANALYSIS")
    print("="*80)
    print()
    
    # Check for duplicate file paths
    print("🔍 Checking for duplicate entries...")
    file_paths = list(processed.keys())
    duplicates = [path for path, count in Counter(file_paths).items() if count > 1]
    if duplicates:
        print(f"   ❌ Found {len(duplicates)} duplicate file paths:")
        for dup in duplicates[:10]:
            print(f"      {dup}")
        if len(duplicates) > 10:
            print(f"      ... and {len(duplicates) - 10} more")
    else:
        print("   ✅ No duplicate file paths found")
    print()
    
    # Check for inconsistencies
    print("🔍 Checking for data inconsistencies...")
    issues = []
    
    for file_path, entry in processed.items():
        # Check if file_path in entry matches the key
        if entry.get('file_path') != file_path:
            issues.append(f"Path mismatch: key='{file_path}', entry.file_path='{entry.get('file_path')}'")
        
        # Check for missing required fields
        if 'status' not in entry:
            issues.append(f"Missing status: {file_path}")
        if 'processed_at' not in entry:
            issues.append(f"Missing processed_at: {file_path}")
        
        # Check for negative values
        if entry.get('duration') is not None and entry['duration'] < 0:
            issues.append(f"Negative duration: {file_path} ({entry['duration']})")
        if entry.get('relationships_count') is not None and entry['relationships_count'] < 0:
            issues.append(f"Negative relationships_count: {file_path} ({entry['relationships_count']})")
    
    if issues:
        print(f"   ⚠️  Found {len(issues)} inconsistencies:")
        for issue in issues[:20]:
            print(f"      {issue}")
        if len(issues) > 20:
            print(f"      ... and {len(issues) - 20} more")
    else:
        print("   ✅ No data inconsistencies found")
    print()
    
    # Analyze by status
    print("📊 Status Analysis:")
    status_groups = defaultdict(list)
    for file_path, entry in processed.items():
        status = entry.get('status', 'unknown')
        status_groups[status].append((file_path, entry))
    
    for status, entries in sorted(status_groups.items()):
        print(f"   {status}: {len(entries)} entries")
        if status == 'completed':
            durations = [e[1].get('duration', 0) for e in entries if e[1].get('duration') is not None]
            rels = [e[1].get('relationships_count', 0) for e in entries if e[1].get('relationships_count') is not None]
            if durations:
                print(f"      Duration: min={min(durations):.1f}s, max={max(durations):.1f}s, avg={sum(durations)/len(durations):.1f}s")
            if rels:
                print(f"      Relationships: min={min(rels)}, max={max(rels)}, avg={sum(rels)/len(rels):.1f}")
    print()
    
    # Check save_in_progress flag
    print("🔍 Checkpoint State:")
    save_in_progress = checkpoint_data.get('save_in_progress', False)
    last_saved_count = checkpoint_data.get('last_saved_count', 0)
    total_count = len(processed)
    
    print(f"   save_in_progress: {save_in_progress}")
    print(f"   last_saved_count: {last_saved_count:,}")
    print(f"   total_entries: {total_count:,}")
    
    if save_in_progress:
        print(f"   ⚠️  WARNING: save_in_progress is True - checkpoint may be in inconsistent state")
    
    if last_saved_count < total_count:
        unsaved = total_count - last_saved_count
        print(f"   ⚠️  WARNING: {unsaved:,} entries may not have been saved to GCS")
    else:
        print(f"   ✅ All entries appear to have been saved")
    print()
    
    # File path patterns
    print("🔍 File Path Patterns:")
    path_patterns = defaultdict(int)
    for file_path in processed.keys():
        # Extract pattern (e.g., form-4-filings/SYMBOL/YEAR/...)
        parts = file_path.split('/')
        if len(parts) >= 2:
            pattern = '/'.join(parts[:2])  # e.g., "form-4-filings/SYMBOL"
            path_patterns[pattern] += 1
    
    print(f"   Found {len(path_patterns)} unique path patterns")
    for pattern, count in sorted(path_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"      {pattern}: {count:,} files")
    if len(path_patterns) > 10:
        print(f"      ... and {len(path_patterns) - 10} more patterns")
    print()


def main():
    checkpoint_path = Path(__file__).parent / CHECKPOINT_FILE
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"📁 Loading: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    detailed_analysis(checkpoint_data)
    
    print("="*80)
    print("✅ DETAILED ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
