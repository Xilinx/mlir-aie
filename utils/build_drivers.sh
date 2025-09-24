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
XDNA_SHA=1efcf88257c2960ee4e82a732edb5799846dad03
XDNA_DIR=xdna-driver
#if [ -d "$XDNA_DIR" ]; then
#    rm -rf $XDNA_DIR
#fi
#git clone https://github.com/amd/xdna-driver.git
export XDNA_SRC_DIR=$(realpath $XDNA_DIR)
cd $XDNA_DIR
echo "Checking out commit $XDNA_SHA..."
#git checkout "$XDNA_SHA"
#git submodule update --init --recursive

echo "Installing XRT dependencies..."
# Install XRT dependencies
cd "$XDNA_SRC_DIR"
#sudo ./tools/amdxdna_deps.sh

echo "Building XRT..."
# Build XRT
cd "$XDNA_SRC_DIR/xrt/build"
#run_build ./build.sh -npu -opt -j 8

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
        base_pattern="*24.04-amd64-base.deb"
        dev_pattern="*24.04-amd64-base-dev.deb"
        runtime_pattern="*24.04-amd64-npu.deb"  # NPU package is the runtime
        ;;
    "24.10")
        base_pattern="*24.10-amd64-base.deb"
        dev_pattern="*24.10-amd64-base-dev.deb"
        runtime_pattern="*24.10-amd64-npu.deb"  # NPU package is the runtime
        ;;
    *)
        echo "Error: Unsupported Ubuntu version ($UBUNTU_VERSION). Supported versions: 24.04, 24.10"
        exit 1
        ;;
esac

# Find matching .deb files
base_pkg=$(ls $base_pattern 2>/dev/null | head -n 1)
dev_pkg=$(ls $dev_pattern 2>/dev/null | head -n 1)
runtime_pkg=$(ls $runtime_pattern 2>/dev/null | head -n 1)

if [[ -z "$base_pkg" || -z "$dev_pkg" || -z "$runtime_pkg" ]]; then
    echo "Error: Could not find required .deb packages:"
    echo " - base:   $base_pattern"
    echo " - dev:    $dev_pattern"
    echo " - runtime: $runtime_pattern"
    exit 1
fi

echo "Installing XRT packages:"
echo " - $base_pkg"
echo " - $dev_pkg"
echo " - $runtime_pkg (NPU runtime)"

# Use dpkg for local .deb files, then fix dependencies if needed
sudo dpkg -i "./$base_pkg" "./$dev_pkg" "./$runtime_pkg" || sudo apt-get -f install -y

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
        # Prefer the new xrt_plugin naming, but also try the older ubuntu*-amdxdna naming
        plugin_patterns=("*24.04-amd64-amdxdna.deb" "ubuntu24.04-x86_64-amdxdna.deb")
        ;;
    "24.10")
        plugin_patterns=("*24.10-amd64-amdxdna.deb" "ubuntu24.10-x86_64-amdxdna.deb")
        ;;
    *)
        echo "Error: Unsupported Ubuntu version ($UBUNTU_VERSION). Supported versions: 24.04, 24.10"
        exit 1
        ;;
esac

# Find the plugin package file by trying multiple patterns
plugin_pkg=""
for pat in "${plugin_patterns[@]}"; do
    match=$(ls $pat 2>/dev/null | head -n 1 || true)
    if [[ -n "$match" ]]; then
        plugin_pkg="$match"
        break
    fi
done

if [[ -z "$plugin_pkg" ]]; then
    echo "Error: Could not find plugin package. Tried patterns: ${plugin_patterns[*]}"
    echo "Directory contents:"
    ls -al
    exit 1
fi
echo "Installing plugin package: $plugin_pkg"
sudo dpkg -i "./$plugin_pkg" || sudo apt-get -f install -y

echo "xdna-driver and XRT built and installed successfully."
echo "Please reboot to apply changes."

