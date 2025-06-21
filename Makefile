# Makefile for C60.ai Executor

install:
	pip install -r requirements.txt

run:
	python executor_agent.py

clean:
	rm -rf output/*

reset:
	git clean -xdf
