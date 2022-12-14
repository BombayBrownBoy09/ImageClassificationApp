install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C,broad-except,bare-except *.py

test:
	python -m pytest -vv test_training.py

all: install format lint test
