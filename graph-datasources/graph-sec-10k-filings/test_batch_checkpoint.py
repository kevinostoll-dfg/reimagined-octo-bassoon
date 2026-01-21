#!/usr/bin/env python3
"""
Test script to verify that batch_process_10k.py creates checkpoints correctly.
Uses the actual functions from batch_process_10k.py.
"""

import sys
import os

# Import the actual functions from batch_process_10k
sys.path.insert(0, os.path.dirname(__file__))
from batch_process_10k import (
    save_checkpoint,
    load_checkpoint,
    update_checkpoint_entry,
    CHECKPOINT_ENABLED,
    CHECKPOINT_GCS_PATH,
    GCS_BUCKET_NAME
)

def test_checkpoint_creation():
    """Test that batch_process_10k.py functions create checkpoints."""
    print("="*80)
    print("TESTING batch_process_10k.py CHECKPOINT CREATION")
    print("="*80)
    print(f"Checkpoint enabled: {CHECKPOINT_ENABLED}")
    print(f"Checkpoint path: {CHECKPOINT_GCS_PATH}")
    print(f"GCS bucket: {GCS_BUCKET_NAME}")
    print()
    
    if not CHECKPOINT_ENABLED:
        print("❌ Checkpoint is disabled. Set CHECKPOINT_ENABLED=true to enable.")
        return False
    
    # Load existing checkpoint (if any)
    print("1. Loading existing checkpoint...")
    checkpoint_data = load_checkpoint()
    initial_count = len(checkpoint_data.get('processed', {}))
    print(f"   Found {initial_count} existing entries")
    print()
    
    # Test adding an entry using the actual batch_process_10k.py function
    print("2. Adding test entry using update_checkpoint_entry()...")
    test_symbol = "TEST"
    test_year = "2025"
    
    # This should call save_checkpoint() internally
    update_checkpoint_entry(
        checkpoint_data,
        symbol=test_symbol,
        year=test_year,
        status="completed",
        duration=99.99,
        error=None
    )
    print(f"   Added {test_symbol}_{test_year}")
    print()
    
    # Load checkpoint again to verify it was saved
    print("3. Reloading checkpoint to verify it was saved...")
    reloaded_data = load_checkpoint()
    reloaded_count = len(reloaded_data.get('processed', {}))
    print(f"   Found {reloaded_count} entries (was {initial_count})")
    
    # Check if our test entry is there
    test_key = f"{test_symbol}_{test_year}"
    if test_key in reloaded_data.get('processed', {}):
        entry = reloaded_data['processed'][test_key]
        print(f"   ✅ Test entry found: {test_key}")
        print(f"      Status: {entry.get('status')}")
        print(f"      Duration: {entry.get('duration')}")
        print(f"      Processed at: {entry.get('processed_at')}")
        
        # Clean up: remove test entry
        print()
        print("4. Cleaning up test entry...")
        checkpoint_data['processed'].pop(test_key, None)
        save_checkpoint(checkpoint_data)
        print(f"   ✅ Removed test entry")
        
        print()
        print("="*80)
        print("✅ SUCCESS: batch_process_10k.py creates checkpoints correctly!")
        print("="*80)
        return True
    else:
        print(f"   ❌ Test entry NOT found: {test_key}")
        print()
        print("="*80)
        print("❌ FAILED: Checkpoint was not saved correctly")
        print("="*80)
        return False

if __name__ == "__main__":
    success = test_checkpoint_creation()
    sys.exit(0 if success else 1)


