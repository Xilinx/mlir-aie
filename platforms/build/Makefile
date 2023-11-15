ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


# make a temporary file and fix up the Build and Test

# Download act and install in temporary location

# Build the local container if it doesn't exist

# checkout a local LLVM if it doesn't exist
test : ${ROOT_DIR}/bin/act

${ROOT_DIR}/bin/act : ${ROOT_DIR}/install.sh
	bash $< -b $(dir $@)

${ROOT_DIR}/install.sh :
	mkdir -p $(dir $@) && curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh -o $(dir $@)/install.sh

all: ${ROOT_DIR}/bin/act
	$< -j build-repo-local \
	--pull=false \
	--workflows ${ROOT_DIR}/buildAndTestLocal.yml \
	-P build-latest=amd/acdcbuild:1.1 \
    --container-options "--volume $(realpath ${ROOT_DIR}/../../)/llvm-central-repo:$(realpath ${ROOT_DIR}/../../)/llvm-central-repo"