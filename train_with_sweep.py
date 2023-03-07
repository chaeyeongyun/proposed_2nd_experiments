import wandb
import argparse
from utils.load_config import get_config_from_json
from u2pl_train import train
sweep_config = {
    'method': 'random',
    'metric':{'goal':'minimize', 'name':'loss'},
    'parameters':{
        # 'epochs':{'min':100, 'max':400},
        'learning_rate':{'distribution':'uniform',
                         'max':0.001,
                         'min':1e-7
            
        }
        
    }
}


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='./config/train/u2pl_train_barlow.json')
opt = parser.parse_args()
cfg = get_config_from_json(opt.config_path)
sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.project_name)
wandb.agent(sweep_id, function=train(cfg), count=20)