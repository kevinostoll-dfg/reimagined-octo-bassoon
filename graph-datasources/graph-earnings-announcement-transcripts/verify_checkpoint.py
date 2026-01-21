#!/usr/bin/env python3
"""
Verification script to test checkpoint PKL creation in GCS.
Tests:
1. Checkpoint file exists in GCS
2. Can download and unpickle the checkpoint
3. Data structure is correct
4. Save/load cycle works correctly
"""

import os
import sys
import pickle
import logging
from typing import Set, Tuple
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# GCS Configuration (matches batch_process_ea.py)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "earnings-announcement-transcripts")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", f"{GCS_BASE_PATH}/.checkpoint/processed_transcripts.pkl")

# Initialize GCS client
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ GCS client initialized")
    logger.info(f"   Bucket: {GCS_BUCKET_NAME}")
    logger.info(f"   Checkpoint path: {CHECKPOINT_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to initialize GCS client: {e}")
    sys.exit(1)


def verify_checkpoint_exists() -> bool:
    """Verify that the checkpoint file exists in GCS."""
    try:
        blob = bucket.blob(CHECKPOINT_PATH)
        exists = blob.exists()
        
        if exists:
            logger.info(f"✅ Checkpoint file exists: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_PATH}")
            
            # Get metadata
            blob.reload()
            size = blob.size
            updated = blob.updated
            logger.info(f"   Size: {size} bytes")
            logger.info(f"   Last updated: {updated}")
            return True
        else:
            logger.warning(f"⚠️  Checkpoint file does NOT exist: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_PATH}")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking checkpoint existence: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def verify_checkpoint_can_be_loaded() -> tuple[bool, Set[Tuple[str, str, str]] | None]:
    """Verify that the checkpoint can be downloaded and unpickled."""
    try:
        blob = bucket.blob(CHECKPOINT_PATH)
        if not blob.exists():
            logger.warning(f"⚠️  Checkpoint does not exist, cannot test loading")
            return False, None
        
        logger.info(f"📥 Downloading checkpoint from GCS...")
        checkpoint_data = blob.download_as_bytes()
        logger.info(f"   Downloaded {len(checkpoint_data)} bytes")
        
        logger.info(f"🔓 Unpickling checkpoint data...")
        processed = pickle.loads(checkpoint_data)
        
        # Verify it's a set
        if not isinstance(processed, set):
            logger.error(f"❌ Checkpoint data is not a set, got {type(processed)}")
            return False, None
        
        # Verify tuple structure
        if processed:
            sample = next(iter(processed))
            if not isinstance(sample, tuple) or len(sample) != 3:
                logger.error(f"❌ Checkpoint tuples have wrong structure. Expected (str, str, str), got {type(sample)} with length {len(sample)}")
                return False, None
            
            # Verify tuple contents are strings
            if not all(isinstance(item, str) for item in sample):
                logger.error(f"❌ Checkpoint tuple contains non-string values: {sample}")
                return False, None
        
        logger.info(f"✅ Checkpoint loaded successfully")
        logger.info(f"   Contains {len(processed)} processed transcript(s)")
        
        # Show sample entries
        if processed:
            logger.info(f"   Sample entries (first 5):")
            for i, entry in enumerate(list(processed)[:5]):
                logger.info(f"      {i+1}. {entry}")
        
        return True, processed
    except pickle.UnpicklingError as e:
        logger.error(f"❌ Failed to unpickle checkpoint data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def test_save_load_cycle() -> bool:
    """Test that we can save and reload a checkpoint correctly."""
    logger.info(f"\n🧪 Testing save/load cycle...")
    
    # Create test data
    test_data: Set[Tuple[str, str, str]] = {
        ("TEST", "2024", "1"),
        ("TEST", "2024", "2"),
        ("AAPL", "2023", "3"),
    }
    
    test_checkpoint_path = f"{GCS_BASE_PATH}/.checkpoint/test_processed_transcripts.pkl"
    
    try:
        # Save test checkpoint
        logger.info(f"💾 Saving test checkpoint to: gs://{GCS_BUCKET_NAME}/{test_checkpoint_path}")
        checkpoint_data = pickle.dumps(test_data)
        blob = bucket.blob(test_checkpoint_path)
        blob.upload_from_string(checkpoint_data, content_type='application/octet-stream')
        
        # Verify it was saved
        blob.reload()
        if not blob.exists():
            logger.error(f"❌ Test checkpoint was not saved (blob.exists() returns False)")
            return False
        
        logger.info(f"   Saved {len(checkpoint_data)} bytes")
        logger.info(f"   Blob size: {blob.size} bytes")
        
        # Load it back
        logger.info(f"📥 Loading test checkpoint back...")
        loaded_data = blob.download_as_bytes()
        loaded_set = pickle.loads(loaded_data)
        
        # Verify data matches
        if loaded_set != test_data:
            logger.error(f"❌ Loaded data does not match saved data!")
            logger.error(f"   Saved: {test_data}")
            logger.error(f"   Loaded: {loaded_set}")
            return False
        
        logger.info(f"✅ Save/load cycle successful - data matches!")
        
        # Clean up test checkpoint
        logger.info(f"🧹 Cleaning up test checkpoint...")
        blob.delete()
        logger.info(f"   Test checkpoint deleted")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in save/load cycle test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to clean up test checkpoint
        try:
            blob = bucket.blob(test_checkpoint_path)
            if blob.exists():
                blob.delete()
                logger.info(f"   Cleaned up test checkpoint")
        except:
            pass
        
        return False


def main():
    """Run all verification tests."""
    print("="*80)
    print("CHECKPOINT VERIFICATION")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Checkpoint Path: {CHECKPOINT_PATH}")
    print()
    
    results = []
    
    # Test 1: Check if checkpoint exists
    print("TEST 1: Checkpoint Existence")
    print("-" * 80)
    exists = verify_checkpoint_exists()
    results.append(("Checkpoint exists", exists))
    print()
    
    # Test 2: Verify checkpoint can be loaded
    print("TEST 2: Checkpoint Loading")
    print("-" * 80)
    can_load, processed_data = verify_checkpoint_can_be_loaded()
    results.append(("Checkpoint can be loaded", can_load))
    print()
    
    # Test 3: Save/load cycle
    print("TEST 3: Save/Load Cycle")
    print("-" * 80)
    cycle_works = test_save_load_cycle()
    results.append(("Save/load cycle works", cycle_works))
    print()
    
    # Summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ All verification tests passed!")
        return 0
    else:
        print("❌ Some verification tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
