SHELL := /bin/bash

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)

STACK ?=

ifeq ($(strip $(STACK)),)
$(error STACK is not set — from the repo root use: make <stack> <target> (stack = directory under engines/ that contains engine.mk))
endif

STACK_DIR := $(REPO_ROOT)/engines/$(STACK)/

ifeq ($(wildcard $(STACK_DIR)engine.mk),)
$(error Unknown stack '$(STACK)': missing $(STACK_DIR)engine.mk)
endif

# Validate required tools are installed
REQUIRED_TOOLS := kubectl gcloud yq envsubst
$(foreach tool,$(REQUIRED_TOOLS),\
  $(if $(shell which $(tool)),,$(error "$(tool)" not found in PATH)))

NAMESPACE ?= default

# Jingra config is engine-local
JINGRA_CONFIG ?= $(STACK_DIR)jingra.yml
JINGRA_SCHEMAS_DIR ?= $(STACK_DIR)schemas
JINGRA_QUERIES_DIR ?= $(STACK_DIR)queries

-include $(REPO_ROOT)/shared/secrets/.secrets.env

include $(STACK_DIR)engine.mk
include $(REPO_ROOT)/shared/make/common.mk
