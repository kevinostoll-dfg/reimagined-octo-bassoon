#!/bin/bash
#
# Script to create a new GCP compute instance with GPU for SEC-BERT benchmarking
#
# Usage:
#   ./create_gpu_node.sh [INSTANCE_NAME] [OPTIONS]
#
# Options:
#   --machine-type TYPE      Machine type (default: n1-standard-16)
#   --gpu-type TYPE          GPU type (default: nvidia-tesla-t4)
#                            Note: T4 works with n1/n2/e2, L4 requires g2, A100 requires a2
#   --gpu-count COUNT        Number of GPUs (default: 1)
#   --zone ZONE              Zone (default: us-central1-a)
#   --project PROJECT        Project ID (default: blacksmith-443107)
#   --disk-size SIZE         Boot disk size in GB (default: 100)
#   --dry-run                Show what would be done without making changes
#
# The script will automatically discover and use available networks in your project.
#

set -euo pipefail

# Default configuration
INSTANCE_NAME="${1:-}"
ZONE="${ZONE:-us-central1-a}"
PROJECT_ID="${PROJECT_ID:-blacksmith-443107}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-16}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"  # T4 works with n1 machine types, L4 requires g2
GPU_COUNT="${GPU_COUNT:-1}"
DISK_SIZE="${DISK_SIZE:-100}"
DRY_RUN="${DRY_RUN:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --machine-type)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --disk-size)
            DISK_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            if [[ -z "${INSTANCE_NAME}" ]]; then
                INSTANCE_NAME="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Generate instance name if not provided
if [[ -z "${INSTANCE_NAME}" ]]; then
    # Generate color-animal-hash name
    colors=("red" "blue" "green" "yellow" "orange" "purple" "pink" "brown" "black" "white" "gray" "silver" "gold" "cyan" "magenta" "lime" "navy" "maroon" "olive" "teal")
    animals=("fox" "bear" "wolf" "eagle" "hawk" "lion" "tiger" "deer" "rabbit" "owl" "cat" "dog" "bird" "fish" "shark" "whale" "dolphin" "seal" "otter" "beaver" "squirrel" "raccoon" "badger" "coyote" "lynx" "bobcat" "panther" "jaguar" "leopard" "cheetah")
    
    color="${colors[$RANDOM % ${#colors[@]}]}"
    animal="${animals[$RANDOM % ${#animals[@]}]}"
    
    chars=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y" "z" "0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
    random_hash=""
    for i in {1..5}; do
        random_hash="${random_hash}${chars[$RANDOM % ${#chars[@]}]}"
    done
    
    INSTANCE_NAME="${color}-${animal}-${random_hash}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI is not installed. Please install it first."
    exit 1
fi

log_section "Creating GPU Compute Instance"

echo "Instance Name: ${INSTANCE_NAME}"
echo "Zone: ${ZONE}"
echo "Project: ${PROJECT_ID}"
echo "Machine Type: ${MACHINE_TYPE}"
echo "GPU: ${GPU_TYPE} x ${GPU_COUNT}"
echo "Boot Disk: ${DISK_SIZE}GB"
echo ""

# Check if instance name already exists
if gcloud compute instances describe "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --quiet >/dev/null 2>&1; then
    log_error "Instance '${INSTANCE_NAME}' already exists in zone '${ZONE}'"
    exit 1
fi

# Verify machine type exists
log_info "Verifying machine type '${MACHINE_TYPE}' in zone '${ZONE}'..."
if ! gcloud compute machine-types describe "${MACHINE_TYPE}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --quiet >/dev/null 2>&1; then
    log_error "Machine type '${MACHINE_TYPE}' not available in zone '${ZONE}'"
    log_info "Available machine types in ${ZONE}:"
    gcloud compute machine-types list \
        --filter="zone:${ZONE}" \
        --project="${PROJECT_ID}" \
        --format="table(name,guestCpus,memoryMb,zone)" \
        | head -20
    exit 1
fi

# Verify GPU type exists
log_info "Verifying GPU type '${GPU_TYPE}' in zone '${ZONE}'..."
if ! gcloud compute accelerator-types describe "${GPU_TYPE}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --quiet >/dev/null 2>&1; then
    log_error "GPU type '${GPU_TYPE}' not available in zone '${ZONE}'"
    log_info "Available GPU types in ${ZONE}:"
    gcloud compute accelerator-types list \
        --filter="zone:${ZONE}" \
        --project="${PROJECT_ID}" \
        --format="table(name,description,zone)"
    exit 1
fi

# Check GPU and machine type compatibility
log_info "Checking GPU and machine type compatibility..."
if echo "${GPU_TYPE}" | grep -q "l4\|a100\|h100"; then
    # L4, A100, H100 require g2 or a2 machine types
    if [[ ! "${MACHINE_TYPE}" =~ ^(g2|a2)- ]]; then
        log_error "GPU type '${GPU_TYPE}' requires g2 or a2 machine types (e.g., g2-standard-8, a2-highgpu-1g)"
        log_error "Current machine type: ${MACHINE_TYPE}"
        log_info "Compatible machine types for ${GPU_TYPE}:"
        log_info "  - g2-standard-8 (8 vCPUs, 32GB RAM) - for L4"
        log_info "  - g2-standard-16 (16 vCPUs, 64GB RAM) - for L4"
        log_info "  - a2-highgpu-1g (12 vCPUs, 85GB RAM, 1x A100) - for A100"
        exit 1
    fi
elif echo "${GPU_TYPE}" | grep -q "t4"; then
    # T4 works with n1, n2, e2 machine types
    if [[ "${MACHINE_TYPE}" =~ ^(g2|a2)- ]]; then
        log_warn "T4 GPU is typically used with n1/n2/e2 machine types, not ${MACHINE_TYPE}"
        log_warn "Consider using nvidia-l4 with g2 machine types for better performance"
    fi
fi
log_info "GPU and machine type compatibility check passed"

# Discover network configuration
NETWORK=""
SUBNET=""
SERVICE_ACCOUNT=""
SCOPES=""
TAGS=""
HAS_EXTERNAL_IP=false

log_info "Discovering network configuration..."

# Extract region from zone (e.g., us-central1-a -> us-central1)
REGION=$(echo "${ZONE}" | sed 's/-[a-z]$//')

# List available networks in the project
log_info "Listing available networks in project..."
AVAILABLE_NETWORKS=$(gcloud compute networks list \
    --project="${PROJECT_ID}" \
    --format="value(name)" 2>/dev/null || echo "")

if [[ -z "${AVAILABLE_NETWORKS}" ]]; then
    log_error "No networks found in project ${PROJECT_ID}."
    log_error "Please create a network first, or ensure you have permissions to list networks."
    exit 1
fi

# Use the first available network (prefer 'default' if it exists)
if echo "${AVAILABLE_NETWORKS}" | grep -q "^default$"; then
    NETWORK="default"
    log_info "Selected network: ${NETWORK} (default network)"
else
    NETWORK=$(echo "${AVAILABLE_NETWORKS}" | head -1)
    log_info "Selected network: ${NETWORK}"
fi

# Try to find a subnet in this region for the network
log_info "Looking for subnets in region ${REGION}..."
AVAILABLE_SUBNETS=$(gcloud compute networks subnets list \
    --network="${NETWORK}" \
    --filter="region:${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(name)" 2>/dev/null || echo "")

if [[ -n "${AVAILABLE_SUBNETS}" ]]; then
    SUBNET=$(echo "${AVAILABLE_SUBNETS}" | head -1)
    log_info "Selected subnet: ${SUBNET}"
else
    log_info "No subnets found in region ${REGION} for network ${NETWORK}. Will use network directly."
fi

# Default configuration: no external IP (use IAP tunnel for SSH)
HAS_EXTERNAL_IP=false
# Default service account will be used (project default)
# Default tags include iap-ssh (added in CREATE_ARGS)

# Default to Debian 12 (same as existing setup scripts)
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
IMAGE_PROJECT="${IMAGE_PROJECT:-debian-cloud}"

# Storage scopes for GCS access
STORAGE_SCOPES="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.full_control"

# Build create command
log_section "Creating Instance"

CREATE_ARGS=(
    "compute" "instances" "create" "${INSTANCE_NAME}"
    "--zone=${ZONE}"
    "--project=${PROJECT_ID}"
    "--machine-type=${MACHINE_TYPE}"
    "--accelerator=type=${GPU_TYPE},count=${GPU_COUNT}"
    "--maintenance-policy=TERMINATE"
    "--image-family=${IMAGE_FAMILY}"
    "--image-project=${IMAGE_PROJECT}"
    "--boot-disk-size=${DISK_SIZE}GB"
    "--boot-disk-type=pd-ssd"
    "--scopes=${STORAGE_SCOPES}"
    "--tags=iap-ssh"
)

# Add network configuration
if [[ -n "${SUBNET}" ]] && [[ "${SUBNET}" != "null" ]] && [[ "${SUBNET}" != "None" ]]; then
    CREATE_ARGS+=("--subnet=${SUBNET}")
elif [[ -n "${NETWORK}" ]] && [[ "${NETWORK}" != "null" ]] && [[ "${NETWORK}" != "None" ]]; then
    CREATE_ARGS+=("--network=${NETWORK}")
fi

# Add service account and scopes (only if specified, otherwise use project default)
# Service account defaults are handled by GCP if not specified

# Add tags (iap-ssh is already in CREATE_ARGS, add any additional tags if specified)
# Default tags (iap-ssh) are already added above

# Handle external IP (default: no external IP, use IAP tunnel)
CREATE_ARGS+=("--no-address")

# Create startup script to install NVIDIA drivers
STARTUP_SCRIPT=$(cat <<'EOFSCRIPT'
#!/bin/bash
set -euo pipefail

# Update system
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl wget gnupg

# Install NVIDIA drivers (for Debian/Ubuntu)
echo "Installing NVIDIA drivers..."
if command -v ubuntu-drivers &> /dev/null; then
    # Ubuntu
    ubuntu-drivers autoinstall || {
        echo "ubuntu-drivers autoinstall failed, trying manual installation..."
        apt-get install -y nvidia-driver-535 nvidia-utils-535 || true
    }
else
    # Debian - install manually
    apt-get install -y linux-headers-$(uname -r) || true
    apt-get install -y nvidia-driver nvidia-smi || {
        echo "Standard nvidia-driver package not available, trying alternative..."
        # Try adding non-free repository for Debian
        apt-get install -y software-properties-common
        add-apt-repository -y contrib non-free || true
        apt-get update -y
        apt-get install -y nvidia-driver nvidia-smi || true
    }
fi

# Verify installation (may not work until after reboot)
echo "Validating NVIDIA driver installation..."
DRIVER_INSTALLED=false
NVIDIA_SMI_WORKS=false

# Check if nvidia-smi command exists
if command -v nvidia-smi &> /dev/null; then
    DRIVER_INSTALLED=true
    echo "✅ nvidia-smi command found"
    
    # Try to run nvidia-smi (may fail until reboot)
    if nvidia-smi &> /dev/null; then
        NVIDIA_SMI_WORKS=true
        echo "✅ nvidia-smi is working!"
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    else
        echo "⚠️  nvidia-smi found but not working yet (requires reboot)"
    fi
else
    echo "⚠️  nvidia-smi command not found"
fi

# Check if NVIDIA kernel modules are loaded
if lsmod | grep -q nvidia; then
    echo "✅ NVIDIA kernel modules are loaded"
    lsmod | grep nvidia | head -5
else
    echo "⚠️  NVIDIA kernel modules not loaded (will load after reboot)"
fi

# Check for NVIDIA device files
if ls /dev/nvidia* &> /dev/null; then
    echo "✅ NVIDIA device files found:"
    ls -1 /dev/nvidia* | head -5
else
    echo "⚠️  NVIDIA device files not found (will appear after reboot)"
fi

echo ""
echo "NVIDIA drivers installation completed."
if [[ "${NVIDIA_SMI_WORKS}" == "true" ]]; then
    echo "✅ GPU is ready to use!"
else
    echo "⚠️  NOTE: GPU access will be fully functional after the instance reboots."
    echo "   After reboot, run: nvidia-smi"
fi

# Create validation script for post-reboot testing
cat > /usr/local/bin/validate-gpu.sh <<'VALIDATION_SCRIPT'
#!/bin/bash
# GPU validation script - run after instance reboot
set -euo pipefail

echo "=========================================="
echo "NVIDIA GPU Validation"
echo "=========================================="
echo ""

ERRORS=0

# Check 1: nvidia-smi command exists
echo "[1/5] Checking nvidia-smi command..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found: $(which nvidia-smi)"
else
    echo "❌ nvidia-smi not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: nvidia-smi works
echo "[2/5] Testing nvidia-smi execution..."
if nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi is working"
    echo "GPU Details:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name driver mem_total mem_free temp; do
        echo "  GPU $idx: $name"
        echo "    Driver: $driver"
        echo "    Memory: ${mem_free}MB / ${mem_total}MB free"
        echo "    Temperature: ${temp}°C"
    done
else
    echo "❌ nvidia-smi failed to execute"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: NVIDIA kernel modules loaded
echo "[3/5] Checking NVIDIA kernel modules..."
if lsmod | grep -q "^nvidia "; then
    echo "✅ NVIDIA kernel modules loaded:"
    lsmod | grep "^nvidia" | head -3
else
    echo "❌ NVIDIA kernel modules not loaded"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: NVIDIA device files
echo "[4/5] Checking NVIDIA device files..."
if ls /dev/nvidia* &> /dev/null; then
    echo "✅ NVIDIA device files found:"
    ls -1 /dev/nvidia* | head -5
else
    echo "❌ NVIDIA device files not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: PyTorch CUDA availability (if Python is available)
echo "[5/5] Checking PyTorch CUDA support..."
if command -v python3 &> /dev/null; then
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null; then
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [[ "${CUDA_AVAILABLE}" == "True" ]]; then
            echo "✅ PyTorch can access CUDA"
            DEVICE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
            if [[ "${DEVICE_COUNT}" -gt 0 ]]; then
                DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
                echo "   Device count: ${DEVICE_COUNT}"
                echo "   GPU 0: ${DEVICE_NAME}"
            fi
        else
            echo "⚠️  PyTorch installed but CUDA not available (may need to install torch with CUDA support)"
        fi
    else
        echo "⚠️  PyTorch not installed or error checking CUDA"
    fi
else
    echo "⚠️  Python3 not available, skipping PyTorch check"
fi
echo ""

# Summary
echo "=========================================="
if [[ "${ERRORS}" -eq 0 ]]; then
    echo "✅ GPU Validation: PASSED"
    echo "   GPU is ready for use!"
    exit 0
else
    echo "❌ GPU Validation: FAILED (${ERRORS} error(s))"
    echo "   Please check the errors above and ensure:"
    echo "   1. Instance has been rebooted after driver installation"
    echo "   2. NVIDIA drivers are properly installed"
    echo "   3. GPU is attached to the instance"
    exit 1
fi
VALIDATION_SCRIPT

chmod +x /usr/local/bin/validate-gpu.sh
echo "✅ Created validation script: /usr/local/bin/validate-gpu.sh"
echo "   Run 'validate-gpu.sh' after reboot to verify GPU is working"
EOFSCRIPT
)

TEMP_SCRIPT=$(mktemp)
echo "${STARTUP_SCRIPT}" > "${TEMP_SCRIPT}"
CREATE_ARGS+=("--metadata-from-file=startup-script=${TEMP_SCRIPT}")

if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "[DRY RUN] Would create instance with:"
    echo "  gcloud ${CREATE_ARGS[*]}"
    rm -f "${TEMP_SCRIPT}"
    exit 0
fi

log_info "Creating instance (this may take a few minutes)..."
log_info "Command: gcloud ${CREATE_ARGS[*]}"

if gcloud "${CREATE_ARGS[@]}"; then
    log_info "Instance created successfully!"
else
    log_error "Failed to create instance"
    rm -f "${TEMP_SCRIPT}"
    exit 1
fi

rm -f "${TEMP_SCRIPT}"

# Get instance details
log_section "Instance Created Successfully"

INSTANCE_INFO=$(gcloud compute instances describe "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --format=json)

EXTERNAL_IP=$(echo "${INSTANCE_INFO}" | jq -r '.networkInterfaces[0].accessConfigs[0].natIP // "None"')
INTERNAL_IP=$(echo "${INSTANCE_INFO}" | jq -r '.networkInterfaces[0].networkIP')

echo "Instance Name: ${INSTANCE_NAME}"
echo "Zone: ${ZONE}"
echo "Machine Type: ${MACHINE_TYPE}"
echo "GPU: ${GPU_TYPE} x ${GPU_COUNT}"
echo "Internal IP: ${INTERNAL_IP}"
echo "External IP: ${EXTERNAL_IP}"
echo ""

log_section "Next Steps"

echo "1. Wait for instance to finish startup (drivers installation may take 5-10 minutes)"
echo ""
echo "2. SSH into the instance:"
if [[ "${EXTERNAL_IP}" == "None" ]]; then
    echo "   gcloud compute ssh ${INSTANCE_NAME} \\"
    echo "       --zone=${ZONE} \\"
    echo "       --project=${PROJECT_ID} \\"
    echo "       --tunnel-through-iap"
else
    echo "   gcloud compute ssh ${INSTANCE_NAME} \\"
    echo "       --zone=${ZONE} \\"
    echo "       --project=${PROJECT_ID}"
fi
echo ""
echo "3. Verify GPU installation (after SSH):"
echo "   # Quick check:"
echo "   nvidia-smi"
echo ""
echo "   # Comprehensive validation:"
echo "   validate-gpu.sh"
echo ""
echo "4. If nvidia-smi doesn't work, reboot the instance:"
echo "   sudo reboot"
echo "   (Wait 2-3 minutes, then SSH back in and run 'validate-gpu.sh')"
echo ""
echo "5. Install Python dependencies and run benchmark:"
echo "   cd graph-datasources/finetune_model"
echo "   python3 benchmark.py --test_csv dev_labeled.csv --device cuda --batch_size 128"
echo ""

log_section "Useful Commands"

echo "Check instance status:"
echo "  gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID} --format='value(status)'"
echo ""
echo "Start instance:"
echo "  gcloud compute instances start ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
echo ""
echo "Stop instance:"
echo "  gcloud compute instances stop ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
echo ""
echo "Delete instance:"
echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"

