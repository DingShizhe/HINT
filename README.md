# Code implementation of the HINT

Code implementation of the paper _Accurate Interpolation for Scattered Data through Hierarchical Residual Refinement_.

The implementation of this work is built upon the foundations of three existing projects: [NIERT](https://github.com/DingShizhe/NIERT), [NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales) and [TFR-HSS-Benchmark](https://github.com/shendu-sw/TFR-HSS-Benchmark).


## Preparation

1. We highly recommend utilizing the conda package manager to create and manage the project's environment:

```bash
conda create -n hint python=3.7
conda activate hint
```

2. Install the required third-party libraries by executing the following command:

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## Construction of Mathit-2D Dataset

The Mathit-2D dataset construction process builds upon the work of [NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales) and [NIERT](https://github.com/DingShizhe/NIERT).

Follow the steps below to create the dataset:

```bash
# generate training equations set
python3 -m src.data.mathit.run_dataset_creation --number_of_equations 1000000 --no-debug

# generate testing equations set
python3 -m src.data.mathit.run_dataset_creation --number_of_equations 150 --no-debug

mkdir -p mathit_data/test_set

# convert the newly created validation dataset in a csv format
python3 -m src.data.mathit.run_dataload_format_to_csv raw_test_path=mathit_data/data/raw_datasets/150

# remove the validation equations from the training set
python3 -m src.data.mathit.run_filter_from_already_existing --data_path mathit_data/data/raw_datasets/1000000 --csv_path mathit_data/test_set/test_nc.csv

python3 -m src.data.mathit.run_apply_filtering --data_path mathit_data/data/raw_datasets/1000000
```

By following these steps, you will generate the Mathit-2D dataset.


## Accessing the PTV and TFRD Dataset

The PTV dataset can be obtained from [here](https://github.com/DingShizhe/PTV-Dataset). This dataset provides valuable resources for interpolating particle velocities and reconstructing velocity fields in velocity-based analyses.

The TFRD datasets can be obtained from [here](https://github.com/shendu-sw/recon-data-generator). This dataset is specifically designed for reconstructing temperature fields from measurements obtained by scattered temperature sensors.



## Training

Follow the steps below to Train HINT:

```bash
# Training on Mathit-2D dataset
CUDA_VISIBLE_DEVICES="0,1" python main.py --config_path ./config/config_Mathit.yml

# Training on Pelrin dataset
CUDA_VISIBLE_DEVICES="0" python main.py --config_path ./config/config_Perlin.yml

# Training on TFRD-ADlet dataset
CUDA_VISIBLE_DEVICES="0,1" python main.py --config_path ./config/config_TFR_adlet.yml

# Training on PTV dataset
CUDA_VISIBLE_DEVICES="0,1" python main.py --config_path ./config/config_PTV.yml
```


## Testing

For Mathit dataset, we certainly need to fix a interpolation task test set from the equation skeleton test set.

```bash
python main.py -m save_Mathit_testdataset_as_file
```

Then we can evaluate HINT on such test set.

```
CUDA_VISIBLE_DEVICES="0,1" python main.py -m test_Mathit --resume_from_checkpoint path_of_hint_checkpoint
```

For evaluation on other datasets, just run:

```
CUDA_VISIBLE_DEVICES="0,1" python main.py -m test_<dataset_name> --resume_from_checkpoint path_of_hint_checkpoint
```
