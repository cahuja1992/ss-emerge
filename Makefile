.PHONY: setup test clean build_docker run_docker

setup:
	pip install -r requirements.txt
	pip install -r test_requirements.txt
	# pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

test:
	pytest tests/

test_unit:
	pytest tests/unit/

test_integration:
	pytest tests/integration/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ build/ dist/

build_docker:
	docker build -t ss-emerge .

run_docker:
	docker run -it --rm --gpus all ss-emerge bash