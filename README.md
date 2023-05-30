# Approximate Probabilistic Query Answering
Probablistic Reasoning over large (incomplete) Knowledge Graphs



## Run on Das 6 Instructions (DEV)
* ENVIRONMENT:
    `conda activate /var/scratch/ybk240/envs/pqa`
* CONFIG: set run variables in default.ini
* DAS:
  `module load prun; module load cuda11.1/toolkit (CHANGE)`
* RUN: `python src/main.py`


## Research Ideas

* Query Optimization (More advanced strategies then: top-k beam search or threshold)
* Scores to probabilities (Tractor etc / different neural structure)
* End to end learning (aProbLog-style backprop)


## Description

Knowledge Graphs are often incomplete. By creating a Knowledge Graphs Embedding (KGE), one can assign a score [0, 1] to every possible triplet <s, r, o> with a neural link predictor (NLP) such as Complex.
The Knowledge Graph is now extended with uncertain facts. Hence, incompletness results in uncertaintity. 
Reasoning over all uncertain facts is not feasiable, hence a form of approximation is required. 


### Approximation
apqa currently supports two strategies:
* Top-k Beam-Search (CQD)
* Threshold

All remaining facts are stored in a Probablistic Database (PDB). Such that the queries can be answered by means of a Datalog Engine (e.g. Glog). A common strategy to rank the answers is Weighted Model Counting (WMC).

## Getting Started

### Dependencies

* Glog
* Glog-Python
* Lib-Kge
* PySDD (WMC)

### Installing

* TODO

### Executing program

* TODO


## Authors

Contributors names and contact info

ex. Yannick Brunink


## License

TODO

## Acknowledgments


* [glog](https://github.com/karmaresearch/glog)
* [cqd](https://github.com/uclnlp/cqd)
