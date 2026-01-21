# GCS Bucket Policy Management

This directory contains scripts for auditing and managing lifecycle and retention policies for the `blacksmith-sec-filings` GCS bucket.

## Scripts

### `audit_and_manage_policies.py`

A comprehensive Python script for auditing and managing GCS bucket lifecycle and retention policies.

#### Features

- **Audit**: View current lifecycle and retention policies
- **Apply Policies**: Set or update lifecycle and retention policies
- **Backup**: Create backups of current policies before making changes
- **Dry Run**: Preview changes without applying them

#### Default Policies

- **Retention Policy**: 30 days (prevents deletion of objects for 30 days)
- **Lifecycle Policy**:
  - Move to Nearline storage after 35 days
  - Move to Coldline storage after 45 days
  - Delete objects after 60 days

#### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Audit current policies (default action)
python audit_and_manage_policies.py

# Audit with explicit flag
python audit_and_manage_policies.py --audit

# Apply default policies (dry run)
python audit_and_manage_policies.py --apply --dry-run

# Apply default policies
python audit_and_manage_policies.py --apply

# Apply custom retention period (60 days)
python audit_and_manage_policies.py --apply --retention-days 60

# Apply custom lifecycle policy from JSON file
python audit_and_manage_policies.py --apply --lifecycle-file custom-policy.json

# Backup current policies
python audit_and_manage_policies.py --backup

# Use different bucket
python audit_and_manage_policies.py --bucket my-bucket-name

# Specify GCP project
python audit_and_manage_policies.py --gcp-project my-project-id
```

#### Command-Line Arguments

- `--bucket`, `-b`: GCS bucket name (default: `blacksmith-sec-filings`)
- `--gcp-project`, `-p`: GCP project ID (default: uses default credentials)
- `--audit`, `-a`: Audit current bucket policies
- `--apply`: Apply lifecycle and retention policies
- `--backup`: Backup current policies to a JSON file
- `--retention-days`, `-r`: Retention period in days (default: 30)
- `--lifecycle-file`, `-l`: Path to JSON file with lifecycle policy
- `--dry-run`: Show what would be done without actually applying changes

#### Lifecycle Policy JSON Format

The lifecycle policy JSON file should follow this format:

```json
{
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
```

Or wrapped in a `lifecycle` object:

```json
{
  "lifecycle": {
    "rule": [
      ...
    ]
  }
}
```

#### Backup Files

Backup files are stored in the `./backups/` directory with the format:
```
YYYY-MM-DD-HHMMSS-{bucket-name}-policies.json
```

Each backup contains:
- Timestamp
- Bucket name and GCP project
- Current retention policy
- Current lifecycle rules
- Full bucket information

#### Examples

**Example 1: Full audit and policy application**

```bash
# 1. Audit current state
python audit_and_manage_policies.py --audit

# 2. Backup current policies
python audit_and_manage_policies.py --backup

# 3. Preview changes
python audit_and_manage_policies.py --apply --dry-run

# 4. Apply policies
python audit_and_manage_policies.py --apply
```

**Example 2: Custom retention period**

```bash
# Apply 90-day retention policy
python audit_and_manage_policies.py --apply --retention-days 90
```

**Example 3: Custom lifecycle policy**

Create a file `my-lifecycle.json`:

```json
{
  "rule": [
    {
      "action": {
        "type": "SetStorageClass",
        "storageClass": "ARCHIVE"
      },
      "condition": {
        "age": 30
      }
    },
    {
      "action": {
        "type": "Delete"
      },
      "condition": {
        "age": 365
      }
    }
  ]
}
```

Then apply it:

```bash
python audit_and_manage_policies.py --apply --lifecycle-file my-lifecycle.json
```

## Authentication

The script uses Google Cloud default credentials. Make sure you have:

1. Installed the Google Cloud SDK
2. Authenticated with `gcloud auth application-default login`
3. Set the appropriate GCP project (or use `--gcp-project`)

## Notes

- **Retention Policy**: Once a retention policy is locked, it cannot be changed. Be careful when applying retention policies.
- **Lifecycle Rules**: Lifecycle rules are applied automatically by GCS based on object age.
- **Storage Classes**: Available storage classes include STANDARD, NEARLINE, COLDLINE, and ARCHIVE (in order of decreasing cost and access speed).

