# Build/test shortcuts for lambdapic.
# Defaults can be overridden on the command line, e.g.:
#   make test JOBS=16 N_MPI=4

JOBS ?= $(shell nproc 2>/dev/null || echo 4)
N_MPI ?= 9

.PHONY: refresh build build_inplace install test test_serial test_mpi clean

refresh: clean build install

build:
	LAMBDAPIC_DISABLE_MPI=1 python -m build

build_inplace:
	python setup.py build_ext --inplace -j$(JOBS)

install:
	pip install .

test: build_inplace
	pytest -n $(JOBS)
	mpiexec -np $(N_MPI) python -m pytest --with-mpi -m mpi -n 0 --no-cov tests/mpi

test_serial: build_inplace
	pytest -n $(JOBS)

test_mpi:
	mpiexec -np $(N_MPI) python -m pytest --with-mpi -m mpi -n 0 --no-cov tests/mpi

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/lambdapic/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf lambdapic.egg-info
	rm -rf src/lambdapic.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf tests/mpi/.coverage
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find src/ -name "*.so" -delete
	find src/ -name "*.h5" -delete
	find src/ -name "*.npz" -delete
	find example/ -name "*.h5" -delete
	find example/ -name "*.npz" -delete
	find example/ -name "*.log" -delete
	find example/ -name "*.json" -delete
	find example/ -type d -name "density" -exec rm -rf {} + 2>/dev/null || true
	find example/ -type d -name "fields" -exec rm -rf {} + 2>/dev/null || true
	find docs/ -name "_build" -type d -exec rm -rf {} + 2>/dev/null || true
	find docs/ -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
