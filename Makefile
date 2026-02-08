# bunlari command kimi run et. 
.PHONY: install train serve test docker-build docker-run rollback

install:
	pip install -e .[dev] || pip install -e .

train:
	housing-train --config configs/train.yaml

serve:
	uvicorn housing_model.service:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t housing-model:latest .

docker-run:
	docker run --rm -p 8000:8000 housing-model:latest

# make serve - run et. sonra postmandan yoxla. copyalar uvicornu . http:.0.0.0:8000/health ; meta
# make docker-build
# docker run --rm -p 8000:8000 housing-model:latest