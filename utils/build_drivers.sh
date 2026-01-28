#!/bin/bash

# Ensure the script exits immediately if a command fails
set -e

# Parse command-line arguments for verbosity
VERBOSE=0
if [[ "$1" == "--verbose" ]]; then
    VERBOSE=1
    echo "Verbose mode enabled."
fi

# Function to run build commands with optional output redirection
run_build() {
    if [[ $VERBOSE -eq 1 ]]; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

echo "Detecting Ubuntu version..."
# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu version detected: $UBUNTU_VERSION"

# Verify Ubuntu version is 22.04 or newer
UBUNTU_MAJOR=$(echo "$UBUNTU_VERSION" | cut -d'.' -f1)
UBUNTU_MINOR=$(echo "$UBUNTU_VERSION" | cut -d'.' -f2)
if [[ "$UBUNTU_MAJOR" -lt 22 ]]; then
    echo "Error: Ubuntu version must be 22.04 or newer. Current version: $UBUNTU_VERSION"
    exit 1
fi

echo "Checking kernel version..."
# Check if the kernel version is at least 6.10
KERNEL_VERSION=$(uname -r | cut -d'.' -f1,2)
REQUIRED_VERSION="6.10"

if [[ "$(echo -e "$KERNEL_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Kernel version must be at least Linux 6.10. Current version: $KERNEL_VERSION"
    exit 1
fi

# For kernel 6.10, verify required kernel config options
if [[ "$KERNEL_VERSION" == "6.10" ]]; then
    echo "Kernel 6.10 detected. Verifying required kernel configuration..."
    
    # Check if kernel config is available
    if [ -f "/boot/config-$(uname -r)" ]; then
        CONFIG_FILE="/boot/config-$(uname -r)"
    elif [ -f "/proc/config.gz" ]; then
        CONFIG_FILE="/proc/config.gz"
    else
        echo "Warning: Cannot find kernel config file. Proceeding anyway..."
        CONFIG_FILE=""
    fi
    
    if [ -n "$CONFIG_FILE" ]; then
        # Check for required config options
        if [ "${CONFIG_FILE##*.}" = "gz" ]; then
            CONFIG_AMD_IOMMU=$(zcat "$CONFIG_FILE" | grep "^CONFIG_AMD_IOMMU=" | cut -d'=' -f2)
            CONFIG_DRM_ACCEL=$(zcat "$CONFIG_FILE" | grep "^CONFIG_DRM_ACCEL=" | cut -d'=' -f2)
        else
            CONFIG_AMD_IOMMU=$(grep "^CONFIG_AMD_IOMMU=" "$CONFIG_FILE" | cut -d'=' -f2)
            CONFIG_DRM_ACCEL=$(grep "^CONFIG_DRM_ACCEL=" "$CONFIG_FILE" | cut -d'=' -f2)
        fi
        
        if [[ "$CONFIG_AMD_IOMMU" != "y" ]]; then
            echo "Error: CONFIG_AMD_IOMMU is not enabled in kernel. This is required for kernel 6.10."
            echo "Please upgrade to kernel 6.11 or newer, or rebuild kernel with CONFIG_AMD_IOMMU=y"
            exit 1
        fi
        
        if [[ "$CONFIG_DRM_ACCEL" != "y" ]]; then
            echo "Error: CONFIG_DRM_ACCEL is not enabled in kernel. This is required for kernel 6.10."
            echo "Please upgrade to kernel 6.11 or newer, or rebuild kernel with CONFIG_DRM_ACCEL=y"
            exit 1
        fi
        
        echo "Kernel 6.10 configuration verified: CONFIG_AMD_IOMMU and CONFIG_DRM_ACCEL are enabled."
    fi
else
    # Verify kernel is at least 6.11
    MINIMUM_RECOMMENDED="6.11"
    if [[ "$(echo -e "$KERNEL_VERSION\n$MINIMUM_RECOMMENDED" | sort -V | head -n1)" != "$MINIMUM_RECOMMENDED" ]]; then
        echo "Warning: Kernel version $KERNEL_VERSION is between 6.10 and 6.11."
        echo "Kernel 6.11 or newer from Ubuntu HWE is recommended for best compatibility."
    else
        echo "Kernel $KERNEL_VERSION detected (>= 6.11). No additional config checks needed."
    fi
fi

echo "Setting up XDNA driver repository..."
# Clone or update the XDNA driver repository and initialize submodules
XDNA_TAG=f3293dc901d733438468cd8b055adb250fa6ced0
if [ -d "xdna-driver" ]; then
    echo "xdna-driver directory already exists. Removing and re-cloning to ensure clean state..."
    rm -rf xdna-driver
fi

echo "Cloning the XDNA driver repository..."
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver
echo "Checking out tag $XDNA_TAG..."
git checkout "$XDNA_TAG"
git submodule update --init --recursive
cd ..
export XDNA_SRC_DIR=$(realpath xdna-driver)

echo "Applying pkg-config fix for CMake compatibility..."
# Fix pkg-config path to use system version instead of old /tools/xgs/bin/pkg-config
sed -i '/# --- PkgConfig ---/a # Force use of system pkg-config to avoid version compatibility issues\nset(PKG_CONFIG_EXECUTABLE "/usr/bin/pkg-config" CACHE FILEPATH "Path to pkg-config executable")' "$XDNA_SRC_DIR/xrt/src/CMake/nativeLnx.cmake"

echo "Applying GCC 13 compatibility fix..."
# Add compiler flag to work around GCC 13 standard library issue
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1 ${CXXFLAGS}"

echo "Installing XRT dependencies..."
# Install XRT dependencies
cd "$XDNA_SRC_DIR"
sudo ./tools/amdxdna_deps.sh

echo "Building XRT..."
# Build XRT
cd "$XDNA_SRC_DIR/xrt/build"
run_build ./build.sh -npu -opt

echo "Removing any existing XRT and XDNA-driver packages..."
# Find and remove any installed packages that start with "xrt"
packages=$(dpkg -l | awk '/^ii/ && $2 ~ /^xrt/ { print $2 }')
if [ -z "$packages" ]; then
    echo "No packages starting with 'xrt' are installed."
else
    echo "Removing the following packages:"
    echo "$packages"
    sudo apt-get remove -y $packages
fi

echo "Installing new XRT packages..."
# Install XRT packages based on Ubuntu version
cd "$XDNA_SRC_DIR/xrt/build/Release"

# Set package patterns based on Ubuntu version
case "$UBUNTU_VERSION" in
    "22.04")
        base_pattern="*22.04-amd64-base.deb"
        dev_pattern="*22.04-amd64-base-dev.deb"
        npu_pattern="*22.04-amd64-npu.deb"
        ;;
    "24.04")
        base_pattern="*24.04-amd64-base.deb"
        dev_pattern="*24.04-amd64-base-dev.deb"
        npu_pattern="*24.04-amd64-npu.deb"
        ;;
    "24.10")
        base_pattern="*24.10-amd64-base.deb"
        dev_pattern="*24.10-amd64-base-dev.deb"
        npu_pattern="*24.10-amd64-npu.deb"
        ;;
    "25.04")
        base_pattern="*25.04-amd64-base.deb"
        dev_pattern="*25.04-amd64-base-dev.deb"
        npu_pattern="*25.04-amd64-npu.deb"
        ;;
    *)
        # For any other version >= 22.04, try to find packages matching the version
        echo "Ubuntu version $UBUNTU_VERSION detected. Attempting to find matching packages..."
        base_pattern="*${UBUNTU_VERSION}-amd64-base.deb"
        dev_pattern="*${UBUNTU_VERSION}-amd64-base-dev.deb"
        npu_pattern="*${UBUNTU_VERSION}-amd64-npu.deb"
        ;;
esac

# Find matching .deb files
base_pkg=$(ls $base_pattern 2>/dev/null | head -n 1)
dev_pkg=$(ls $dev_pattern 2>/dev/null | head -n 1)
npu_pkg=$(ls $npu_pattern 2>/dev/null | head -n 1)
if [[ -z "$base_pkg" || -z "$dev_pkg" || -z "$npu_pkg" ]]; then
    echo "Error: Could not find .deb packages matching one or more patterns:"
    echo " - $base_pattern"
    echo " - $dev_pattern"
    echo " - $npu_pattern"
    echo "Available packages:"
    ls -1 *.deb 2>/dev/null || echo "No .deb files found"
    exit 1
fi
echo "Installing packages: $base_pkg, $dev_pkg, $npu_pkg"
sudo apt reinstall -y "./$base_pkg"
sudo apt reinstall -y "./$dev_pkg"
sudo apt reinstall -y "./$npu_pkg"

echo "Building XDNA Driver..."
# Build XDNA Driver
cd "$XDNA_SRC_DIR/build"
run_build ./build.sh -release

echo "Installing XDNA plugin..."
# Install XDNA plugin based on Ubuntu version
cd "$XDNA_SRC_DIR/build/Release"

# Set plugin pattern based on Ubuntu version
case "$UBUNTU_VERSION" in
    "22.04")
        plugin_pattern="*22.04-amd64-amdxdna.deb"
        ;;
    "24.04")
        plugin_pattern="*24.04-amd64-amdxdna.deb"
        ;;
    "24.10")
        plugin_pattern="*24.10-amd64-amdxdna.deb"
        ;;
    "25.04")
        plugin_pattern="*25.04-amd64-amdxdna.deb"
        ;;
    *)
        # For any other version >= 22.04, try to find package matching the version
        echo "Ubuntu version $UBUNTU_VERSION detected. Attempting to find matching plugin package..."
        plugin_pattern="*${UBUNTU_VERSION}-amd64-amdxdna.deb"
        ;;
esac

# Find the plugin package file
plugin_pkg=$(ls $plugin_pattern 2>/dev/null | head -n 1)
if [[ -z "$plugin_pkg" ]]; then
    echo "Error: Could not find plugin package matching pattern: $plugin_pattern"
    echo "Available packages:"
    ls -1 *.deb 2>/dev/null || echo "No .deb files found"
    exit 1
fi
echo "Installing plugin package: $plugin_pkg"
sudo apt reinstall -y "./$plugin_pkg"

echo "xdna-driver and XRT built and installed successfully."
