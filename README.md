# NAIAD

NAIAD is a tool developed by Neptune Bio for modeling and predicting phenotypic outcomes of combinatorial genetic perturbations. We use an active learning framework that efficiently discovers optimal gene pairs that induced desired cellular phenotypes, leveraging knowledge from single-gene perturbation effects and learned gene embeddings. Evaluated on four CRISPR combinatorial perturbation datasets totaling over 350,000 genetic interactions, NAIAD, trained on a small dataset, outperforms existing models by up to 40% relative to the second-best.

Installation:
- Set up a development environment with python>=3.8
- Clone GitHub repo locally and install NAIAD with `pip install -e .`