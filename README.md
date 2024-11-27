# wildfire_cmf
This repo uses HPE CMF to track ML artifacts and code for wildfire detection

# Installation of CMF
```shell
conda create -n wildfire_cmf python=3.10 
conda activate wildfire_cmf
git clone https://github.com/HewlettPackard/cmf.git
pip install ./cmf

#Directory tree of this repo
This repo has 3 main folders namely artifacts, training, inferencing.
The artifacts folder contains the source data and generated artifacts that to be pushed to/pulled from cmf server.
The training folder contains the scripts to perform data collection, train/test split, learning rate finder, training, evaluation.
At each stage, cmf API will be called to log artifacts, models, metrics.
The inferencing folder contains the scripts to perfrom inferencing on live/simulated cameras feeds. At this stage, cmf API will be called to log inferencing result
