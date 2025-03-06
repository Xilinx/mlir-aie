#!/bin/bash

# Ensure the script exits immediately if a command fails
set -e

# Check if the kernel version is at least 6.11
KERNEL_VERSION=$(uname -r | cut -d'.' -f1,2)
REQUIRED_VERSION="6.11"

if [[ "$(echo -e "$KERNEL_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Kernel version must be at least Linux 6.11. Current version: $KERNEL_VERSION"
    exit 1
fi

# Clone the XDNA driver repository and initialize submodules
git clone https://github.com/amd/xdna-driver.git
export XDNA_SRC_DIR=$(realpath xdna-driver)
cd xdna-driver
git submodule update --init --recursive

# Install XRT dependencies
cd "$XDNA_SRC_DIR"
sudo ./tools/amdxdna_deps.sh

# Build XRT
cd "$XDNA_SRC_DIR/xrt/build"
./build.sh -npu -opt

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)

# Install XRT packages based on Ubuntu version
cd "$XDNA_SRC_DIR/xrt/build/Release"
case "$UBUNTU_VERSION" in
    "24.04")
        sudo apt reinstall ./xrt_202510.2.19.0_24.04-amd64-base.deb
        sudo apt reinstall ./xrt_202510.2.19.0_24.04-amd64-base-dev.deb
        ;;
    "24.10")
        sudo apt reinstall ./xrt_202510.2.19.0_24.10-amd64-base.deb
        sudo apt reinstall ./xrt_202510.2.19.0_24.10-amd64-base-dev.deb
        ;;
    *)
        echo "Error: Unsupported Ubuntu version ($UBUNTU_VERSION). Supported versions: 24.04, 24.10"
        exit 1
        ;;
esac

# Build XDNA Driver
cd "$XDNA_SRC_DIR/build"
./build.sh -release
./build.sh -package

# Install XDNA plugin based on Ubuntu version
cd "$XDNA_SRC_DIR/build/Release"
case "$UBUNTU_VERSION" in
    "24.04")
        sudo apt reinstall ./xrt_plugin.2.19.0_ubuntu24.04-x86_64-amdxdna.deb
        ;;
    "24.10")
        sudo apt reinstall ./xrt_plugin.2.19.0_ubuntu24.10-x86_64-amdxdna.deb
        ;;
esac

echo "xdna-driver and XRT built and installed."
echo "Please reboot to apply changes."

