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

echo "Checking kernel version..."
# Check if the kernel version is at least 6.11
KERNEL_VERSION=$(uname -r | cut -d'.' -f1,2)
REQUIRED_VERSION="6.11"

if [[ "$(echo -e "$KERNEL_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Kernel version must be at least Linux 6.11. Current version: $KERNEL_VERSION"
    exit 1
fi

echo "Cloning the XDNA driver repository..."
# Clone the XDNA driver repository and initialize submodules
XDNA_SHA=e9d2788a884784e3531e95d65b923c2252a1132e
git clone https://github.com/amd/xdna-driver.git
export XDNA_SRC_DIR=$(realpath xdna-driver)
cd xdna-driver
echo "Checking out commit $XDNA_SHA..."
git checkout "$XDNA_SHA"
git submodule update --init --recursive

echo "Installing XRT dependencies..."
# Install XRT dependencies
cd "$XDNA_SRC_DIR"
sudo ./tools/amdxdna_deps.sh

echo "Building XRT..."
# Build XRT
cd "$XDNA_SRC_DIR/xrt/build"
run_build ./build.sh -npu -opt

echo "Detecting Ubuntu version..."
# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Ubuntu version detected: $UBUNTU_VERSION"

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
case "$UBUNTU_VERSION" in
    "24.04")
        base_suffix="24.04-amd64-base.deb"
        dev_suffix="24.04-amd64-base-dev.deb"
        runtime_suffix="24.04-amd64-Runtime.deb"
        ;;
    "24.10")
        base_suffix="24.10-amd64-base.deb"
        dev_suffix="24.10-amd64-base-dev.deb"
        runtime_suffix="24.10-amd64-Runtime.deb"
        ;;
    *)
        echo "Error: Unsupported Ubuntu version ($UBUNTU_VERSION). Supported versions: 24.04, 24.10"
        exit 1
        ;;
esac

# Find matching .deb files
base_pkg=$(ls *"$base_suffix" 2>/dev/null | head -n 1)
dev_pkg=$(ls *"$dev_suffix" 2>/dev/null | head -n 1)
runtime_pkg=$(ls *"$runtime_suffix" 2>/dev/null | head -n 1)
if [[ -z "$base_pkg" || -z "$dev_pkg" || -z "$runtime_pkg" ]]; then
    echo "Error: Could not find .deb packages matching one or more suffixes:"
    echo " - $base_suffix"
    echo " - $dev_suffix"
    echo " - $runtime_suffix"
    exit 1
fi
sudo apt reinstall "./$base_pkg"
sudo apt reinstall "./$dev_pkg"
sudo apt reinstall "./$runtime_pkg"

echo "Building XDNA Driver..."
# Build XDNA Driver
cd "$XDNA_SRC_DIR/build"
run_build ./build.sh -release
run_build ./build.sh -package

echo "Installing XDNA plugin..."
# Install XDNA plugin based on Ubuntu version
cd "$XDNA_SRC_DIR/build/Release"
case "$UBUNTU_VERSION" in
    "24.04")
        plugin_suffix="24.04-amd64-amdxdna.deb"
        ;;
    "24.10")
        plugin_suffix="24.10-amd64-amdxdna.deb"
        ;;
    *)
        echo "Error: Unsupported Ubuntu version ($UBUNTU_VERSION). Supported versions: 24.04, 24.10"
        exit 1
        ;;
esac

# Find the plugin package file
plugin_pkg=$(ls *"$plugin_suffix" 2>/dev/null | head -n 1)
if [[ -z "$plugin_pkg" ]]; then
    echo "Error: Could not find plugin package matching suffix: $plugin_suffix"
    exit 1
fi
sudo apt reinstall "./$plugin_pkg"

echo "xdna-driver and XRT built and installed successfully."
echo "Please reboot to apply changes."

