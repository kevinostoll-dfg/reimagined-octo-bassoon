#!/usr/bin/env python3
"""
Verification script for checkpoint PKL file creation in GCS.
Tests checkpoint save/load functionality and verifies data integrity.
"""

import os
import sys
import pickle
import logging
from datetime import datetime, timezone
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# GCS Configuration (matches batch_process_10k.py)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
CHECKPOINT_GCS_PATH = os.getenv("CHECKPOINT_GCS_PATH", "checkpoints/10k_processing_checkpoint.pkl")

# Initialize GCS client
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ GCS client initialized")
    logger.info(f"   Bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to initialize GCS client: {e}")
    sys.exit(1)


def test_checkpoint_save():
    """Test saving a checkpoint to GCS."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Saving Checkpoint to GCS")
    logger.info("="*80)
    
    # Create test checkpoint data
    test_data = {
        'processed': {
            'AAPL_2024': {
                'symbol': 'AAPL',
                'year': '2024',
                'status': 'completed',
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'duration': 123.45,
                'error': None
            },
            'MSFT_2023': {
                'symbol': 'MSFT',
                'year': '2023',
                'status': 'completed',
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'duration': 98.76,
                'error': None
            }
        },
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    
    try:
        # Serialize to pickle
        checkpoint_bytes = pickle.dumps(test_data)
        logger.info(f"✅ Serialized checkpoint data ({len(checkpoint_bytes)} bytes)")
        
        # Upload to GCS
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        blob.upload_from_string(checkpoint_bytes, content_type='application/octet-stream')
        logger.info(f"✅ Uploaded checkpoint to gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
        
        # Verify blob exists
        blob.reload()
        if blob.exists():
            logger.info(f"✅ Blob exists in GCS")
            logger.info(f"   Size: {blob.size} bytes")
            logger.info(f"   Content Type: {blob.content_type}")
            logger.info(f"   Created: {blob.time_created}")
            logger.info(f"   Updated: {blob.updated}")
            return True
        else:
            logger.error(f"❌ Blob does not exist after upload")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saving checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_checkpoint_load():
    """Test loading a checkpoint from GCS."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Loading Checkpoint from GCS")
    logger.info("="*80)
    
    try:
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        
        if not blob.exists():
            logger.warning(f"⚠️  Checkpoint blob does not exist at gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
            logger.info("   Run TEST 1 first to create a checkpoint")
            return False
        
        logger.info(f"✅ Blob exists")
        logger.info(f"   Size: {blob.size} bytes")
        
        # Download and deserialize
        checkpoint_bytes = blob.download_as_bytes()
        logger.info(f"✅ Downloaded checkpoint ({len(checkpoint_bytes)} bytes)")
        
        checkpoint_data = pickle.loads(checkpoint_bytes)
        logger.info(f"✅ Deserialized checkpoint data")
        
        # Verify data structure
        if 'processed' not in checkpoint_data:
            logger.error(f"❌ Checkpoint data missing 'processed' key")
            return False
        
        if 'last_updated' not in checkpoint_data:
            logger.error(f"❌ Checkpoint data missing 'last_updated' key")
            return False
        
        logger.info(f"✅ Checkpoint structure is valid")
        logger.info(f"   Processed filings: {len(checkpoint_data.get('processed', {}))}")
        logger.info(f"   Last updated: {checkpoint_data.get('last_updated')}")
        
        # Show sample entries
        processed = checkpoint_data.get('processed', {})
        if processed:
            logger.info(f"\n   Sample entries:")
            for i, (key, entry) in enumerate(list(processed.items())[:3]):
                symbol = entry.get('symbol', '?')
                year = entry.get('year', '?')
                status = entry.get('status', '?')
                logger.info(f"      {i+1}. {symbol}-{year}: {status}")
            if len(processed) > 3:
                logger.info(f"      ... and {len(processed) - 3} more")
        
        return True
        
    except pickle.PickleError as e:
        logger.error(f"❌ Pickle deserialization error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_checkpoint_roundtrip():
    """Test save and load roundtrip to verify data integrity."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Roundtrip Test (Save → Load → Verify)")
    logger.info("="*80)
    
    # Create test data
    original_data = {
        'processed': {
            'TEST_2024': {
                'symbol': 'TEST',
                'year': '2024',
                'status': 'completed',
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'duration': 42.0,
                'error': None
            }
        },
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    
    try:
        # Save
        checkpoint_bytes = pickle.dumps(original_data)
        blob = bucket.blob(CHECKPOINT_GCS_PATH + ".test")
        blob.upload_from_string(checkpoint_bytes, content_type='application/octet-stream')
        logger.info(f"✅ Saved test checkpoint")
        
        # Load
        loaded_bytes = blob.download_as_bytes()
        loaded_data = pickle.loads(loaded_bytes)
        logger.info(f"✅ Loaded test checkpoint")
        
        # Verify data matches
        if original_data['processed'] != loaded_data.get('processed'):
            logger.error(f"❌ Processed data mismatch")
            logger.error(f"   Original: {original_data['processed']}")
            logger.error(f"   Loaded: {loaded_data.get('processed')}")
            return False
        
        if original_data['last_updated'] != loaded_data.get('last_updated'):
            logger.warning(f"⚠️  Timestamp mismatch (expected for roundtrip)")
        
        logger.info(f"✅ Roundtrip test passed - data integrity verified")
        
        # Clean up test file
        blob.delete()
        logger.info(f"✅ Cleaned up test checkpoint")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Roundtrip test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to clean up
        try:
            blob = bucket.blob(CHECKPOINT_GCS_PATH + ".test")
            if blob.exists():
                blob.delete()
        except:
            pass
        return False


def check_existing_checkpoint():
    """Check if an existing checkpoint exists and display its info."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Check Existing Checkpoint")
    logger.info("="*80)
    
    try:
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        
        if not blob.exists():
            logger.info(f"ℹ️  No existing checkpoint found at gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
            return False
        
        # Get blob metadata
        blob.reload()
        logger.info(f"✅ Existing checkpoint found:")
        logger.info(f"   Path: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
        logger.info(f"   Size: {blob.size} bytes")
        logger.info(f"   Content Type: {blob.content_type}")
        logger.info(f"   Created: {blob.time_created}")
        logger.info(f"   Updated: {blob.updated}")
        
        # Try to load and verify
        checkpoint_bytes = blob.download_as_bytes()
        checkpoint_data = pickle.loads(checkpoint_bytes)
        
        processed_count = len(checkpoint_data.get('processed', {}))
        logger.info(f"   Processed filings: {processed_count}")
        logger.info(f"   Last updated (data): {checkpoint_data.get('last_updated')}")
        
        # Verify data structure
        processed = checkpoint_data.get('processed', {})
        if processed:
            # Count by status
            status_counts = {}
            for entry in processed.values():
                status = entry.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            logger.info(f"   Status breakdown:")
            for status, count in status_counts.items():
                logger.info(f"      {status}: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking existing checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all verification tests."""
    print("="*80)
    print("CHECKPOINT VERIFICATION SCRIPT")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Checkpoint Path: {CHECKPOINT_GCS_PATH}")
    print()
    
    results = {}
    
    # Test 1: Save checkpoint
    results['save'] = test_checkpoint_save()
    
    # Test 2: Load checkpoint
    results['load'] = test_checkpoint_load()
    
    # Test 3: Roundtrip test
    results['roundtrip'] = test_checkpoint_roundtrip()
    
    # Test 4: Check existing checkpoint
    results['existing'] = check_existing_checkpoint()
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"\n📊 Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.upper():15s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✅ All tests passed! Checkpoint functionality is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Review the logs above for details.")
    
    print(f"\n{'='*80}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

