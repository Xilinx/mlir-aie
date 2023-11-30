     FROM ubuntu:20.04
     ARG DEBIAN_FRONTEND=noninteractive
     ENV TZ=Etc/UTC
     RUN apt-get -qq update && \
          apt-get -qq install -y git nodejs pip ninja-build clang lld zstd && \
          pip install cmake numpy psutil pybind11 rich