import yaml
from yaml import Loader



def get_options(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt





