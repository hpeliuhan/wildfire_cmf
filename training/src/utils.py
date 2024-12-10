import os
from cmflib import cmf
def is_graph_enabled():
    graph_env = os.getenv("NEO4J", "True")
    return graph_env.lower() == "true"

def set_cmf_environment(filepath, pipeline_name):
    graph = is_graph_enabled()
    
    metawriter = cmf.Cmf(filepath=filepath, pipeline_name=pipeline_name, graph=graph)
    return metawriter