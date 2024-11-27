#!/bin/bash
#Function to check if a folder exists and create it if it does not
check_and_create_folder() {
    local folder_path=$1
    if [ ! -d "$folder_path" ]; then
        echo "Folder '$folder_path' does not exist. Creating it..."
        mkdir -p "$folder_path"
    else
        echo "Folder '$folder_path' already exists."
    fi
}

#check if there is artifacts parent folder
ART_PARENT="artifacts"
check_and_create_folder $ART_PARENT

# check if there is artifacts/labeldata folder
LABLEDATA_FOLDER="artifacts/labeldata"
check_and_create_folder LABLEDATA_FOLDER


#Download the data to the datasource folder, fill your own code
datasource=/data/cmf_sage/sage-smoke-detection/training/artifacts/labeldata/all.tar.gz
cp $datasource $LABLEDATA_FOLDER

################################
# run data loading scripts
################################

# check if there is artifacts/loaded folder
DATA_LOADED_FOLDER="artifacts/loaded"
check_and_create_folder $DATA_LOADED_FOLDER
printf "\n[1/5] [RUNNING DATALOADING STEP         ]\n"
python training/src/data_collection.py $LABLEDATA_FOLDER $DATA_LOADED_FOLDER

################################
# run train test data split
################################

# check if there is artifacts/split_data folder
SPLIT_DATA_FOLDER="artifacts/split_data"
check_and_create_folder $SPLIT_DATA_FOLDER
printf "\n[2/5] [RUNNING TRAIN TEST SLIT STEP     ]\n"
#python training/src/train_test_split.py $DATA_LOADED_FOLDER $SPLIT_DATA_FOLDER

################################
# final metadata/artifact push
################################

printf "CMF METADATA PUSH"
cmf metadata push -p WILDFIRE -f cmf

printf "CMF ARTIFACT PUSH"
cmf artifact push -p WILDFIRE -f cmf


