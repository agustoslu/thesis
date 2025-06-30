# thesis
Differentially Private NLP for Medical Records

## Installation
```bash
git clone https://github.com/agustoslu/thesis
cd thesis/dpnlp_lib
pip install -e .
```

## How to Run
You can run multiple experiments in parallel using Slurm with the following command.
```bash
python main.py -m task=mortality,phenotyping data_split=iid,non-iid,dirichlet_moderate algorithm=dp-sgd, dp-ftrl
```
This would launch 12 jobs using different degrees of data heterogeneity using specified algorithms.

## Folder Structure For Hydra-Config Files
```bash
thesis/
├── main.py                           
├── conf/                             
│   ├── config.yaml                   
│   ├── task/                         
│   │   ├── mortality.yaml            
│   │   └── phenotyping.yaml          
│   ├── data_split/                   
│   │   ├── iid.yaml
│   │   ├── non_iid.yaml
│   │   └── dirichlet_moderate.yaml
│   ├── algorithm/                    
│   │   ├── dp_sgd.yaml
│   │   └── dp_ftrl.yaml
│   └── hydra/
│       └── launcher/
│           └── submitit_slurm.yaml  
```
## Data Maps
Data maps used when calling DataManager in builder.py can be obtained [here](https://github.com/YerevaNN/mimic3-benchmarks/tree/v1.0.0-alpha/mimic3benchmark/resources).
