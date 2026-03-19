PYTHON = python
CONFIG = configs/base.yaml

.PHONY: all dataset baseline behavioral analysis activations probing directions steering mmlu figures clean

# === Full Pipeline ===
all: figures

# === Individual Steps ===
dataset:
	$(PYTHON) -m src.dataset.run --config $(CONFIG)

baseline: dataset
	$(PYTHON) -m src.baseline.run --config $(CONFIG)

behavioral: baseline
	$(PYTHON) -m src.behavioral.run --config $(CONFIG)

analysis: behavioral
	$(PYTHON) -m src.analysis.run --config $(CONFIG)

activations: behavioral
	$(PYTHON) -m src.activations.run --config $(CONFIG)

probing: activations
	$(PYTHON) -m src.probing.run --config $(CONFIG)

directions: probing
	$(PYTHON) -m src.directions.run --config $(CONFIG)

steering: directions
	$(PYTHON) -m src.steering.run --config $(CONFIG)

mmlu: steering
	$(PYTHON) -m src.mmlu.run --config $(CONFIG)

figures: analysis directions steering mmlu
	$(PYTHON) -m src.figures.run --config $(CONFIG)

# === Single Model Shortcuts ===
# Usage: make single-model MODEL=llama3_8b
single-model:
	$(PYTHON) -m src.baseline.run --config $(CONFIG) --model $(MODEL)
	$(PYTHON) -m src.behavioral.run --config $(CONFIG) --model $(MODEL)
	$(PYTHON) -m src.activations.run --config $(CONFIG) --model $(MODEL)
	$(PYTHON) -m src.probing.run --config $(CONFIG) --model $(MODEL)
	$(PYTHON) -m src.steering.run --config $(CONFIG) --model $(MODEL)
	$(PYTHON) -m src.mmlu.run --config $(CONFIG) --model $(MODEL)

# === Utilities ===
score-only:
	$(PYTHON) -m src.behavioral.run --config $(CONFIG) --score-only

annotations:
	$(PYTHON) -m src.behavioral.run --config $(CONFIG) --generate-annotations

permtest:
	$(PYTHON) -m src.probing.run --config $(CONFIG) --permtest

clean:
	rm -rf outputs/
