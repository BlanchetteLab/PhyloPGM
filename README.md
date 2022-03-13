# PhyloPGM

## Requirements

## TFBS Prediction Problem

### Data Preparation

### Train FactorNet

### FactorNet Predictions
`python FactorNet/predict_factornet.py toy-data/chipseq-ortho-val.csv toy-data/model-human-FOXA1.pth toy-data/pred-chipseq-ortho-val.csv`

`python FactorNet/predict_factornet.py toy-data/chipseq-ortho-test.csv toy-data/model-human-FOXA1.pth toy-data/pred-chipseq-ortho-test.csv`

### Compute PhyloPGM Scores
`python PhyloPGM/create_PhyloPGM_input.py toy-data/pred-chipseq-ortho-val.csv toy-data/df-pred-chipseq-ortho-val.csv PhyloPGM/tree.pkl 1000`

`python PhyloPGM/create_PhyloPGM_input.py toy-data/pred-chipseq-ortho-test.csv toy-data/df-pred-chipseq-ortho-test.csv PhyloPGM/tree.pkl 1000`

`python PhyloPGM/run_PhyloPGM.py toy-data/df-pred-chipseq-ortho-val.csv toy-data/df-pred-chipseq-ortho-test.csv PhyloPGM/tree.pkl toy-data/df-pgm-100`


## RNA-RBP Binding Prediction Problem

### Data Preparation

### Train RNATracker

### RNATracker Predictions
`python RNATracker/predict_rnatracker.py toy-data/ortho-val-100 toy-data/base_model.pth toy-data/pred-ortho-val-100`

`python RNATracker/predict_rnatracker.py toy-data/ortho-test-100 toy-data/base_model.pth toy-data/pred-ortho-test-100`

### Compute PhyloPGM Scores
`python PhyloPGM/create_PhyloPGM_input.py toy-data/pred-ortho-val-100 toy-data/df-pred-ortho-val-100 PhyloPGM/tree.pkl 1000`

`python PhyloPGM/create_PhyloPGM_input.py toy-data/pred-ortho-test-100 toy-data/df-pred-ortho-test-100 PhyloPGM/tree.pkl 1000`

`python PhyloPGM/run_PhyloPGM.py toy-data/df-pred-ortho-val-100 toy-data/df-pred-ortho-test-100 PhyloPGM/tree.pkl toy-data/df-pgm-100`

## Putative Functional Sites Analysis





