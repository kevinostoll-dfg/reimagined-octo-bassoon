#!/bin/bash
#
# Standalone GPU validation script
# Validates that NVIDIA drivers are properly installed and working on the current node
#
# Usage:
#   ./validate_gpu.sh
#   or
#   bash validate_gpu.sh
#
# Exit codes:
#   0 - All validations passed
#   1 - One or more validations failed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "$1"
}

# Track errors
ERRORS=0
GPU_DETECTED=false

print_header "NVIDIA GPU Validation"

# Preliminary check: Detect if GPU hardware is present
print_info "Detecting GPU hardware..."
if command -v lspci &> /dev/null; then
    if lspci | grep -i "nvidia\|vga.*3d" &> /dev/null; then
        GPU_DETECTED=true
        print_success "GPU hardware detected via lspci:"
        lspci | grep -i "nvidia\|vga.*3d" | while read -r line; do
            echo "  $line"
        done
    else
        print_warning "No NVIDIA GPU detected via lspci"
        print_info "   This instance may not have a GPU attached"
    fi
else
    print_warning "lspci not available, cannot detect GPU hardware"
fi

# Check for GPU in /sys/class/drm (display devices)
if ls /sys/class/drm/card*/device/vendor 2>/dev/null | head -1 | xargs cat 2>/dev/null | grep -q "0x10de"; then
    GPU_DETECTED=true
    print_success "NVIDIA GPU detected in /sys/class/drm"
fi

# For GCP: Check instance metadata (if available)
if command -v curl &> /dev/null; then
    GCP_METADATA=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/attributes/accelerator-type" 2>/dev/null || echo "")
    if [[ -n "${GCP_METADATA}" ]]; then
        GPU_DETECTED=true
        print_success "GPU detected in GCP instance metadata: ${GCP_METADATA}"
    fi
fi

if [[ "${GPU_DETECTED}" == "false" ]]; then
    print_warning "No GPU hardware detected on this instance"
    print_info "   If you expected a GPU, verify:"
    print_info "   - Instance was created with --accelerator flag"
    print_info "   - GPU is attached: gcloud compute instances describe INSTANCE --zone=ZONE --format='get(guestAccelerators)'"
    print_info "   - For GCP: Use create_gpu_node.sh to create a GPU instance"
    echo ""
fi
echo ""

# Check 1: nvidia-smi command exists
print_info "[1/6] Checking nvidia-smi command..."
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_SMI_PATH=$(which nvidia-smi)
    print_success "nvidia-smi found: ${NVIDIA_SMI_PATH}"
else
    print_error "nvidia-smi not found"
    print_info "   Install NVIDIA drivers to get nvidia-smi"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: nvidia-smi works
print_info "[2/6] Testing nvidia-smi execution..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        print_success "nvidia-smi is working"
        print_info "GPU Details:"
        # Get GPU information
        if nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx name driver mem_total mem_free temp; do
            # Trim whitespace
            idx=$(echo "$idx" | xargs)
            name=$(echo "$name" | xargs)
            driver=$(echo "$driver" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            mem_free=$(echo "$mem_free" | xargs)
            temp=$(echo "$temp" | xargs)
            echo "  GPU $idx: $name"
            echo "    Driver: $driver"
            echo "    Memory: ${mem_free}MB / ${mem_total}MB free"
            echo "    Temperature: ${temp}°C"
        done; then
            : # Success
        else
            # Fallback: simple nvidia-smi output
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -3 || true
        fi
    else
        print_error "nvidia-smi failed to execute"
        print_info "   This usually means:"
        print_info "   - Drivers are not properly installed"
        print_info "   - Instance needs to be rebooted after driver installation"
        print_info "   - GPU is not attached to the instance"
        ERRORS=$((ERRORS + 1))
    fi
else
    print_warning "Skipping nvidia-smi test (command not found)"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: NVIDIA kernel modules loaded
print_info "[3/6] Checking NVIDIA kernel modules..."
if lsmod | grep -q "^nvidia "; then
    print_success "NVIDIA kernel modules loaded:"
    lsmod | grep "^nvidia" | head -5 | while read -r line; do
        echo "  $line"
    done
else
    print_error "NVIDIA kernel modules not loaded"
    print_info "   Expected modules: nvidia, nvidia_uvm, nvidia_drm, etc."
    print_info "   This usually means drivers need to be installed or instance needs reboot"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: NVIDIA device files
print_info "[4/6] Checking NVIDIA device files..."
if ls /dev/nvidia* &> /dev/null 2>&1; then
    print_success "NVIDIA device files found:"
    ls -1 /dev/nvidia* 2>/dev/null | head -5 | while read -r device; do
        if [[ -c "$device" ]]; then
            echo "  $device (character device)"
        else
            echo "  $device"
        fi
    done
    DEVICE_COUNT=$(ls -1 /dev/nvidia* 2>/dev/null | wc -l)
    print_info "   Total NVIDIA devices: ${DEVICE_COUNT}"
else
    print_error "NVIDIA device files not found"
    print_info "   Expected: /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm, etc."
    print_info "   These appear after drivers are loaded (usually after reboot)"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: PyTorch CUDA availability (if Python is available)
print_info "[5/6] Checking PyTorch CUDA support..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 || echo "unknown")
    print_info "   Python found: ${PYTHON_VERSION}"
    
    # Check if torch is installed
    if python3 -c "import torch" &> /dev/null 2>&1; then
        print_info "   PyTorch is installed"
        
        # Check CUDA availability
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [[ "${CUDA_AVAILABLE}" == "True" ]]; then
            print_success "PyTorch can access CUDA"
            DEVICE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
            if [[ "${DEVICE_COUNT}" -gt 0 ]]; then
                DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
                CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
                print_info "   CUDA version: ${CUDA_VERSION}"
                print_info "   Device count: ${DEVICE_COUNT}"
                print_info "   GPU 0: ${DEVICE_NAME}"
                
                # Test a simple CUDA operation
                if python3 -c "import torch; x = torch.randn(1).cuda(); print('CUDA tensor test: OK')" &> /dev/null 2>&1; then
                    print_success "CUDA tensor operations working"
                else
                    print_warning "CUDA tensor operations test failed"
                fi
            else
                print_warning "PyTorch reports CUDA available but device count is 0"
            fi
        else
            print_warning "PyTorch installed but CUDA not available"
            print_info "   This may mean:"
            print_info "   - PyTorch was installed without CUDA support (CPU-only)"
            print_info "   - CUDA drivers are not properly configured"
            print_info "   - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        fi
    else
        print_info "   PyTorch not installed"
        print_info "   Install with: pip install torch"
        print_info "   For CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    fi
else
    print_info "   Python3 not available, skipping PyTorch check"
fi
echo ""

# Summary
print_header "Validation Summary"

if [[ "${ERRORS}" -eq 0 ]]; then
    print_success "GPU Validation: PASSED"
    print_info "   All checks passed! GPU is ready for use."
    echo ""
    print_info "You can now use CUDA in your applications:"
    print_info "   python3 benchmark.py --device cuda --batch_size 128"
    exit 0
else
    print_error "GPU Validation: FAILED (${ERRORS} error(s))"
    echo ""
    
    if [[ "${GPU_DETECTED}" == "false" ]]; then
        print_info "⚠️  No GPU hardware detected on this instance"
        echo ""
        print_info "If you need a GPU instance:"
        print_info "1. Create a new GPU instance using:"
        print_info "   cd compute_node && ./create_gpu_node.sh [INSTANCE_NAME]"
        echo ""
        print_info "2. Or verify current instance has GPU attached:"
        print_info "   gcloud compute instances describe $(hostname 2>/dev/null || echo 'INSTANCE_NAME') \\"
        print_info "       --zone=$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/zone' 2>/dev/null | xargs basename || echo 'ZONE') \\"
        print_info "       --format='get(guestAccelerators)'"
        echo ""
    else
        print_info "GPU hardware detected but drivers not working"
        echo ""
        print_info "Troubleshooting steps:"
        print_info "1. Verify GPU is attached to the instance:"
        INSTANCE_NAME=$(hostname 2>/dev/null || echo "INSTANCE_NAME")
        ZONE=$(curl -s -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/zone' 2>/dev/null | xargs basename || echo "ZONE")
        print_info "   gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(guestAccelerators)'"
        echo ""
        print_info "2. Install NVIDIA drivers:"
        print_info ""
        print_info "   # RECOMMENDED: Use GCP's official installation script (works for Debian/Ubuntu):"
        print_info "   curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py | sudo python3"
        print_info ""
        print_info "   # Alternative for Debian (requires non-free repository):"
        print_info "   sudo apt-get update"
        print_info "   sudo apt-get install -y linux-headers-\$(uname -r)"
        print_info "   # Add non-free repository if not already added:"
        print_info "   sudo sed -i 's/main$/main contrib non-free/' /etc/apt/sources.list"
        print_info "   sudo apt-get update"
        print_info "   sudo apt-get install -y nvidia-driver nvidia-smi"
        print_info ""
        print_info "   # Alternative for Ubuntu:"
        print_info "   sudo ubuntu-drivers autoinstall"
        echo ""
        print_info "3. After installing drivers, reboot the instance:"
        print_info "   sudo reboot"
        echo ""
        print_info "4. Check driver installation:"
        print_info "   dpkg -l | grep nvidia"
        echo ""
        print_info "5. Check system logs for NVIDIA errors (requires sudo):"
        print_info "   sudo dmesg | grep -i nvidia | tail -20"
        print_info "   sudo journalctl -u nvidia-persistenced -n 50"
        echo ""
        print_info "6. For GCP instances, see official docs:"
        print_info "   https://cloud.google.com/compute/docs/gpus/install-drivers-gpu"
    fi
    exit 1
fi

