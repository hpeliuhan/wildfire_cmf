import click
import requests
import os 
from urllib.parse import urlparse
from cmflib import cmf
from utils import is_graph_enabled, set_cmf_environment
import yaml

def get_filename_from_url(url):
    """
    Extract the filename from the URL.
    """
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)

def get_filename_from_response(response):
    """
    Extract the filename from the Content-Disposition header in the HTTP response.
    """
    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        filename = content_disposition.split('filename=')[-1].strip('"')
        return filename
    return None

def download(config_file:str, dir_config_file: str) -> None:
    execution_config=yaml.safe_load(open(config_file))["download"]
    url=execution_config["url"]
    execution_dir_config=yaml.safe_load(open(dir_config_file))["dir_config"]
    download_dir=execution_dir_config["download"]["output"]

    print(url)
    print(download_dir)
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    filename=get_filename_from_response(response) or get_filename_from_url(url)
    output_path = os.path.join(download_dir, filename)
    # Create the output directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {url} to {output_path}")

    meta_writer = set_cmf_environment("cmf", "WILDFIRE")
    _ = meta_writer.create_context(pipeline_stage="data_collection")
    _ = meta_writer.create_execution(execution_type="download", custom_properties={"url": url})
    _ = meta_writer.log_dataset(output_path, "output")



@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def download_cli(config_file:str, dir_config_file: str) -> None:
    download(config_file, dir_config_file)


if __name__ == '__main__':
    download_cli()