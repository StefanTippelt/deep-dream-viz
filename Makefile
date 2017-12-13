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
<<<<<<< HEAD
		-v ${PWD}/src/:/opt/deepdream/src/ \
=======
		-v ${PWD}/src/img/:/opt/deepdream/src/img/ \
>>>>>>> 8ee3e24cfa31473c5d60e82da241d7b9d5d854db
		${DOCKER_IMAGE}
