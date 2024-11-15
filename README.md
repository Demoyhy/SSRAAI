we present an innovative model that learns sequence and structural representations to predict antibody-antigen interactions 
(SSRAAI). We extracted structural features by constructing contact maps from predicted PDB 3D structures.

The predicted PDB file is stored in the SSRAAI/Data folder.

### Installation
```bash
pip install -r requirements.txt
```

### Training
Before training, all the files (```*.7z```) in the ```dataset/corpus/processed_mat/``` need to be decompressed. These decompressed files are processed datasets which are the input of the models.

Execute the following scripts to train antigen-antibody neutralization model with kmer features on the HIV dataset.
```bash
python model_trainer/ssraai_kmer_embedding_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody neutralization model with kmer and pssm features on the HIV dataset.
```bash
python model_trainer/ssraai_kmer_pssm_embedding_cls_trainer.py --mode train
```

Execute the following scripts to train antigen-antibody neutralization model on the SARS-CoV-2 dataset.
```bash
python model_trainer/ssraai_kmer_embedding_cov_cls_trainer.py --mode train
```




### Preprocessing dataset
The data pre-processing module is in the folder of ```processing/```. There are three sub-folders in the processing folder, ```hiv_cls```, ```hiv_reg```, and ```cov_cls```. The pre-processing can be understood by following the scripts of processing.py as well as the *.py under processing/hiv_cls/, processing/hiv_reg/, and processing/cov_cls/.

- ```processing/hiv_cls/``` includes the scripts and the source data that generate the dataset for HIV classification. The source file is ```dataset_hiv_cls6.xlsx```, which contains four fields ```antibody_seq```, ```virus_seq```, ```label```, and ```split```. ```processing/hiv_cls/corpus/cls``` contains data indices. The generated dataset for HIV classification will be under ```processing/hiv_cls/corpus/processed_mat```. 
- ```processing/cov_cls/``` includes the scripts and the source data that generate the dataset for SARS-CoV2 classification. The source file is ```dataset_cov_cls6.xlsx```, which contains four fields ```antibody_seq```, ```virus_seq```, ```label```, and ```split```. ```processing/cov_cls/corpus/cov_cls``` contains data indices. The generated dataset for SARS-CoV2 will be under ```processing/cov_cls/corpus/processed_mat```. 



The pssm needs to be obtained from the POSSUM ([Wang, J. et al. Possum: a bioinformatics toolkit for generating numerical sequence feature descriptors based on pssm profiles. Bioinformatics 33, 2756–2758 2017](https://academic.oup.com/bioinformatics/article/33/17/2756/3813283)) and placed in the pssm folder. We select the Uniref50 database to generate PSSMs. 

Execute the following scripts to process the HIV dataset for classification.
```bash
python processing/hiv_cls/processing.py
```

Execute the following scripts to process the SARS-CoV-2 dataset.
```bash
python processing/cov_cls/processing.py
```

There is no additional criteria for filtering the data of the other datasets. For more details of data collection and features, please see the in the subsections of **Data** and **Feature** of the **Method** section in Page 9 ~ 10 of the manuscript.



