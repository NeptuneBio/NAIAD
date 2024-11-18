# NAIAD: Active learning for efficient discovery of optimal gene combinations in the combinatorial perturbation space

 NAIAD is a novel active learning framework from Neptune Bio designed to model phenotypic outcomes of combinatorial genetic perturbations and guide experimental design. It focuses on two key goals:
 1. **Small sample learning**  
   Achieve high prediction performance with limited initial experimental data.

 2. **Efficient recommendation system for experiments**  
   Recommend gene pairs that maximize information gain, accelerating convergence with fewer AI + experimental iterations.

By optimizing experimental resources, NAIAD reduces the need for exhaustive testing and efficiently identifies gene combinations that drive desired cellular phenotypes.

### Installation 
- Set up a development environment with python>=3.8
- Clone GitHub repo locally and install NAIAD with `pip install -e .`