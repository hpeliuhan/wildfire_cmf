# Wildfire_CMF
This repo uses HPE CMF to track ML artifacts and code for wildfire detection

## Directory tree of this repo
This repo has 3 main folders namely artifacts, training, inferencing.
The artifacts folder contains the source data and generated artifacts that to be pushed to/pulled from cmf server.
The training folder contains the scripts to perform data collection, train/test split, learning rate finder, training, evaluation.
At each stage, cmf API will be called to log artifacts, models, metrics.
The inferencing folder contains the scripts to perfrom inferencing on live/simulated cameras feeds. At this stage, cmf API will be called to log inferencing result.

## getting Started 
Follow below step to setup virtual environment for CMF development.

#### Prerequsit
Users will need to setup CMF server as prerequisit. 
To setup CMF server remotely, users can follow this instruction:
#### Installation
Users may edit the cmf_init.sh script to fill in the cmf server detail.
```shell
git clone https://github.com/HewlettPackard/cmf.git
pip install ./cmf
git clone https://github.com/hpeliuhan/wildfire_cmf.git
cd wildfire_cmf
conda create -n wildfire_cmf python=3.10 
conda activate wildfire_cmf
pip install -r requirements.txt
./cmf_init.sh
```
#### Training
