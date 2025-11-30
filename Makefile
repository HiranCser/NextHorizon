PYTHON ?= python3
VENV_PYTHON ?= $(PYTHON)

.PHONY: all precompute index clean

all: precompute index

precompute:
	@echo "Running precompute (features + embeddings)"
	./scripts/run_precompute.sh

index:
	@echo "Building ANN index (FAISS)"
	./scripts/build_index.sh

clean:
	@echo "Cleaning generated artifacts (this removes tfidf, features, embeddings, index)"
	rm -f build_training_dataset/*.features.csv build_training_dataset/*.tfidf.* build_training_dataset/*.emb.* build_training_dataset/faiss.index build_training_dataset/*.report.json
