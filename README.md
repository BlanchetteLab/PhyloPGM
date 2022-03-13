# PhyloPGM

## Requirements

## TFBS Prediction Problem

### Data Preparation

### Train FactorNet

### FactorNet Predictions

### Compute PhyloPGM Scores

## RNA-RBP Binding Prediction Problem

### Data Preparation

### Train RNATracker

### RNATracker Predictions
`python RNATracker/predict_rnatracker.py toy-data/ortho-val-100 toy-data/base_model.pth toy-data/pred-ortho-val-100`

`python RNATracker/predict_rnatracker.py toy-data/ortho-test-100 toy-data/base_model.pth toy-data/pred-ortho-test-100`

### Compute PhyloPGM Scores
`python PhyloPGM/create_PhyloPGM_input.py toy-data/pred-ortho-val-100 toy-data/pred-ortho-val-100 PhyloPGM/tree.pkl 1000`

`python PhyloPGM/create_PhyloPGM_input.py toy-data/pred-ortho-test-100 toy-data/pred-ortho-test-100 PhyloPGM/tree.pkl 1000`

`python PhyloPGM/run_PhyloPGM.py toy-data/df-pred-ortho-val-100 toy-data/df-pred-ortho-test-100 PhyloPGM/tree.pkl toy-data/df-pgm-100`

## Putative Functional Sites Analysis





