# Build and push: make image=docker.elastic.co/employees/sachinfrayne/jingra:1
# See RESTORE.md in elasticsearch-gcp/ if this file was reverted.
TAG := ${image}

all: build pull

build:
	@echo "\n::: Building and pushing multi-arch image $(TAG)"
	docker buildx build --push -f Dockerfile --platform linux/amd64,linux/arm64/v8 --tag $(TAG) .

# Use if you see "parent snapshot does not exist" (stale/corrupt cache)
build-no-cache:
	@echo "\n::: Building and pushing (no cache) $(TAG)"
	docker buildx build --no-cache --push -f Dockerfile --platform linux/amd64,linux/arm64/v8 --tag $(TAG) .

clean:
	docker buildx prune -f

pull:
	docker pull $(TAG)
