# NAIAD: Active learning for efficient discovery of optimal gene combinations in the combinatorial perturbation space

 NAIAD is a novel active learning framework from Neptune Bio designed to model phenotypic outcomes of combinatorial genetic perturbations and guide experimental design. It focuses on two key goals:
 1. **Small sample learning**  
   Achieve high prediction performance with limited initial experimental data.

 2. **Efficient recommendation system for experiments**  
   Recommend gene pairs that maximize information gain, accelerating convergence with fewer AI + experimental iterations.

By optimizing experimental resources, NAIAD reduces the need for exhaustive testing and efficiently identifies gene combinations that drive desired cellular phenotypes.
<p align="center"><img src="https://github.com/NeptuneBio/NAIAD/blob/main/img/naiad_framework.png" alt="naiad" width="900px" /></p>

### Installation 
- Set up a development environment with python>=3.8
- Clone GitHub repo locally and install NAIAD with `pip install -e .`


### Tutorials

| Name | Description |
|-----------------|-------------|
| [Dataset Creation](tutorials/data_preparation_tutorial.ipynb) | Tutorial on how to create customized combinatorial perturbation data|
| [Model Training](tutorials/naiad_tutorial.ipynb) | Tutorial on how to train NAIAD |
| [Experiments Recommendation](tutorials/naiad_tutorial.ipynb) | Tutorial on how to recommend gene pairs for follow-up experiments |


### Cite Us

