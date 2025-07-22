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

Instead of sequential runs you can also batch them together to make better use of GPU, which is the recommended approach for Mortality and Phenotyping Tasks with the current configuration since the architecture used and FL setting does not make heavy use of GPU.
```bash
python batch_runner.py
```

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

## Troubleshooting
Hydra will shorten error messages for readability but sometimes it causes the root of issue cannot be seen. You can set an environment variable ```HYDRA_FULL_ERROR=1``` to see full stack trace.

To debug config resolution:

```bash
python main.py task=mortality algorithm=dp-sgd --cfg job --resolve
```
