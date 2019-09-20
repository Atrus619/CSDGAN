### Set up docker-compose container

# https://gist.github.com/mpneuried/0594963ad38e68917ef189b4e6a269db

# import deploy config
env ?= .env
include $(env)
export $(shell sed 's/=.*//' $(env))

# DOCKER TASKS
build: ## Build the container locally
	sudo docker build -t csdgan:latest .

up: ## Build the container. If issue about address already in use, run sudo nginx -s stop
	sudo nginx -s stop
	sudo docker-compose up -d

serve: ## Serve the container on ngrok
	sudo ngrok http $(APP_BIND_PORT)

reset_nginx: ## Kills nginx process if port is in use
	sudo nginx -s stop

stop:  ## Stops container
	sudo docker kill csdgan
	sudo docker kill redis
	sudo docker kill mysql
	sudo docker rm csdgan
	sudo docker rm redis
	sudo docker rm mysql
	sudo docker rm rq-worker

install_ngrok:  ## Installs ngrok
	sudo snap install ngrok

# local dev
dev_down: ## shut down app for local development, along with dependencies
	redis-cli shutdown

worker_up: ## start worker for app for local development
	rq worker CSDGAN

# HELP
.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

