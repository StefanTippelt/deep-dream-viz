TAG ?= local-dev

DOCKER_IMAGE := stefan/deepdream-viz:${TAG}

.PHONY: build
build:
	docker build -t ${DOCKER_IMAGE} .

.PHONY: run
run:
	docker run -it \
		-e FOO=${BAR} \
		-p 9999:9999 \
		-v ${PWD}/src/:/opt/deepdream/src/ \
		${DOCKER_IMAGE}
