# Tools to build locally

This directory contains tools that all the project to be built locally using the github workflow. 

This is achieved using nektos/act - a utility that allows github actions to be executed in local docker instances. 

# Dependencies

Docker ( configured with unprivileged access using 'docker' group)

# How to use

To start the docker cache service ( compatible with the ccache github action)

    make cacheup

To build the local container used for build and test

    make container

To run a local build ( with ccache on all subsequent runs)

    make all