# Vault Operations for graph-sec-F4-filings

This directory contains scripts to manage environment variables in HashiCorp Vault.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Vault token:
```bash
export VAULT_TOKEN="your-vault-token-here"
```

## Usage

### Push Environment Variables to Vault

The `push_to_vault.py` script will:
1. Parse all Python scripts in the project to find `os.getenv()` calls
2. Load variables from `env` file if it exists
3. Collect current environment variable values
4. Create a snapshot and push to Vault

**Dry run (preview what would be pushed):**
```bash
python push_to_vault.py --dry-run
```

**Actually push to Vault:**
```bash
python push_to_vault.py
```

## Vault Configuration

- **Mount Point**: `blacksmith-project-secrets` (KV v2)
- **Secret Path**: `graph-sec-F4-filings`
- **Vault URL**: `https://vault.zagreus.deerfieldgreen.com`

## Environment Variables

The script automatically discovers all environment variables used in:
- `batch_process_f4.py`
- `process_document.py`
- `download_to_gcs.py`

**Runtime variables excluded** (these are set at runtime, not stored as secrets):
- `PYTHON_EXECUTABLE`

**All other variables are required** and will be stored in Vault. If a variable is not set, it will be stored as an empty string and you may need to set it manually in the Vault UI.

## Vault UI

After pushing, view the secrets in the Vault UI:
https://vault.zagreus.deerfieldgreen.com/ui/vault/secrets/blacksmith-project-secrets/show/graph-sec-F4-filings
