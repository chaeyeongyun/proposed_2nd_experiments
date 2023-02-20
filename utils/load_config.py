from easydict import EasyDict
import json

def get_config_from_json(jsonfile):
    with open(jsonfile, 'r') as f:
        try:
           config_dict =  json.load(f) 
           config = EasyDict(config_dict)
           return config
        except ValueError:
           print("INVALID JSON file format.. Please provide a good json file")
           exit(-1)