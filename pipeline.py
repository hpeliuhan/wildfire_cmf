import yaml
import subprocess
import os
from cmflib import cmf
from training.src.utils import set_cmf_environment
import requests

class Pipeline:
    def __init__(self, config_file):
        self.config_file = config_file
        self.params = self.load_params()
        self.dir_config_file = "training/dir_config.yaml"

    def set_environment(self):
        self.filepath=self.params['pipeline']['filepath']
        self.filename=self.params['pipeline']['filename']
        self.metawriter = set_cmf_environment(self.filepath,self.filename)
        


    def load_params(self):
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def run_execution(self, stage_name,execution_name,execution_params):
        print(f"=== Running {execution_name} in {stage_name} ===")
        print(f"{execution_name} params:", execution_params)
        script_path = execution_params['script_path']
        config_path = execution_params.get('config_path', '')
        dir_config_path = execution_params.get('dir_config_path', self.dir_config_file)
        
        # Construct the command
        command = ["python", script_path]
        if config_path:
            command.append(config_path)
        command.append(dir_config_path)
        
        subprocess.run(command)
        print(f"=== {execution_name} executed successfully ===")
    

    def run_stage(self, stage_name):
        stage_params = self.params['pipeline']['stages'][stage_name]
        for execution_name, execution_params  in stage_params['executions'].items():
            self.run_execution(stage_name,execution_name,execution_params)

    def run(self):
        print("=== Loading pipeline Config file:", self.config_file, " ===")
        print("=== Params loaded successfully ===")
        print("There are", len(self.params['pipeline']['stages']), "pipeline stages:", self.params['pipeline']['stages'].keys())
        # Run each stage
        for stage_name in self.params['pipeline']['stages']:
            self.run_stage(stage_name)

    def upload_to_cmf(self):
        self.result=cmf.artifact_push(self.filename,self.filepath)
        self.result=cmf.metadata_push(self.filename,self.filepath)



    def test_connection(self,address):
        #url=address
        # URL to the cmf-server Docker container
        try:
            response = requests.get(address)  # Send a GET request
            if response.status_code == 200:
                print("Connection successful. Server is up and running.")
            else:
                print(f"Unexpected response code: {response.status_code}")
                print("Connection failed.")
        except requests.ConnectionError as e:
            print(f"Failed to establish connection: {e}")

# Call the function to test the connection


if __name__ == "__main__":
    pipeline = Pipeline("pipeline.yaml")
    pipeline.set_environment()
    pipeline.test_connection("http://192.168.30.116:3000")
    pipeline.run()
    pipeline.upload_to_cmf()