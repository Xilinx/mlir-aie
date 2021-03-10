include ../config.make

export BUILD_DIR := $(shell readlink -f ../build$(BUILD)/aie)
export BUILD_CROSS_DIR := $(shell readlink -f ../build$(BUILD)-cross/aie)

${BUILD_DIR}/.dir:
	mkdir -p $(dir $@)
	touch $@

${BUILD_CROSS_DIR}/.dir:
	mkdir -p $(dir $@)
	touch $@

# This should build a libLLVM-10git.so which we can link against easily.
build: ${BUILD_DIR}/.dir
	echo `pwd`; ./config.sh ${BUILD_DIR} . ${INSTALL_DIR}; cd ${BUILD_DIR}; ninja

build-cross: ${BUILD_CROSS_DIR}/.dir
	echo `pwd`; ./config-cross.sh ${BUILD_CROSS_DIR} . ${INSTALL_DIR}-cross; cd ${BUILD_CROSS_DIR}; ninja

doc: ${BUILD_DIR}/.dir
	echo `pwd`; ./config.sh ${BUILD_DIR} . ${INSTALL_DIR}; cd ${BUILD_DIR}; ninja mlir-doc

install: ${BUILD_DIR}/.dir
	echo `pwd`; ./config.sh ${BUILD_DIR} . ${INSTALL_DIR}; cd ${BUILD_DIR}; ninja install

install-cross: ${BUILD_CROSS_DIR}/.dir
	echo `pwd`; ./config-cross.sh ${BUILD_CROSS_DIR} . ${INSTALL_DIR}-cross; cd ${BUILD_CROSS_DIR}; ninja install

test:
	cd ${BUILD_DIR}; ninja check-aie
