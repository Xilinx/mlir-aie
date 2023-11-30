ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LOCAL_LLVM_REPO:=$(realpath ${ROOT_DIR}/../../)/llvm-central-repo

export ACT_CACHE_AUTH_KEY=$(shell id -u --name)-cache-server

# make a temporary file and fix up the Build and Test

#  the local container if it doesn't exist


# Create a directory for a local LLVM if it doesn't exist

# Download act and install in temporary location
test : ${ROOT_DIR}/bin/act

${ROOT_DIR}/bin/act : ${ROOT_DIR}/install.sh
	bash $< -b $(dir $@)

${ROOT_DIR}/install.sh :
	mkdir -p $(dir $@) && curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh -o $(dir $@)/install.sh

${LOCAL_LLVM_REPO} :
	mkdir -p $@

checkip:
	@docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${ACT_CACHE_AUTH_KEY}-app-1


container : ${ROOT_DIR}/Dockerfile
	docker build -t amd/acdcbuild:1.1 ${ROOT_DIR}

all: ${ROOT_DIR}/bin/act ${LOCAL_LLVM_REPO}
	export ACTIONS_IP=`docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${ACT_CACHE_AUTH_KEY}-app-1` && \
	$< -C ${ROOT_DIR}/../.. -j build-repo-local \
	--pull=false \
	--workflows ${ROOT_DIR}/buildAndTestLocal.yml \
	-P build-latest=amd/acdcbuild:1.1 \
    --container-options "--volume ${LOCAL_LLVM_REPO}:${LOCAL_LLVM_REPO}" \
	--env ACTIONS_CACHE_URL=http://$${ACTIONS_IP}:8080/   \
	--env ACTIONS_RUNTIME_URL=http://$${ACTIONS_IP}:8080/ \
	--env ACTIONS_RUNTIME_TOKEN=${ACT_CACHE_AUTH_KEY} \
	--network="${ACT_CACHE_AUTH_KEY}_default"

# What is the value of this cache server? 
# - Compatibility with github
# - Sharing with other developers. 

${ROOT_DIR}/cache-server : 
	mkdir -p $@

${ROOT_DIR}/cache-server/docker-compose.yml: 
	git clone git@github.com:sp-ricard-valverde/github-act-cache-server.git $(dir $@) 

.PHONY : cacheup
cacheup : ${ROOT_DIR}/cache-server/docker-compose.yml
	docker compose -f $<  --project-name ${ACT_CACHE_AUTH_KEY} up --build --detach

cachedown:  ${ROOT_DIR}/cache-server/docker-compose.yml
	docker compose -f $<  --project-name ${ACT_CACHE_AUTH_KEY} down 


