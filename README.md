# Hybrid Molecule Featurization
This repository contains the code used to integrate RDKit chemical domain descriptors with GNN-encoded features for improved performance on MoleculeNet classification and regression benchmarks.


---
Our Conda environment can be replicated using the <i>'environment.yml'</i> file -  
```bash
conda env create -f environment.yml -n <environment name>
conda activate <environment name>
```

  
To prepare the MoleculeNet dataset files for training, for RDKit featurization, the following script can be executed as follows -  
```bash
python ./featurization.py --dataset './Data/<benchmark>'
```




The GCN and GGNN encoders can be trained and their respective embeddings can be used to train the downstream classifiers by executing one of two python scripts, as demonstrated for the BACE benchmark with default parameters and paths -

```bash
python GCN_Feature_Predictions.py \  # Or GGNN_Feature_Predictions.py
  --epochs 10 \                      # Number of epochs to train
  --lr 0.001 \                       # Initial learning rate
  --folder Data/BACE \               # Path to the benchmark dataset folder
  --hidden 512 \                     # Number of hidden units
  --dropout 0.5 \                    # Dropout rate
  --batch_size 128 \                 # Batch size
  --natom 58 \                       # Number of node features
  --benchmark bace \                 # Benchmark dataset name
  --nclass 1 \                       # Number of output classes
  --type classification              # Task type: classification or regression
```   

The output performance scores will be stored in a new CSV file generated by the script.
