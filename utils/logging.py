import torch 
import wandb
from typing import List
import os

#TODO: 개선/ 추가
class Logger():
    def __init__(self, cfg, logger_name):
        wandb.init(project=cfg.project_name,
                   name=logger_name
                #    notes="baseline",
                #    tags = ["csp+unet+cutmix"]
        )
        # config setting
        self.config_dict = dict()
        for key in cfg.train.wandb_config:
            self.config_dict[key] = cfg.train[key]
        
        for i in cfg.train.wandb_metrics:
            if i in ["loss"]:
                wandb.define_metric(i, summary='min')
            if i in ["miou"]:
                wandb.define_metric(i, summary='max')
        
        # initialize log dict
        self.log_dict = dict()
        for key in cfg.train.wandb_log:
            self.log_dict[key] = None
            
    def logging(self, epoch):
        wandb.log(self.log_dict, step=epoch)
    
    def config_update(self):
        wandb.config.update(self.config_dict, allow_val_change=True)
    
    # def end(self, summary_dict):
    #     for key, value in summary_dict.items():
    #         wandb.run.summary[key] = value

def save_ckpoints(model_1, model_2, epoch, batch_idx, optimizer_1, optimizer_2, filepath):
    torch.save({'model_1':model_1,
               'model_2':model_2,
               'epoch':epoch,
               'batch_idx':batch_idx,
               'optimizer_1':optimizer_1,
               'optimizer_2':optimizer_2}, filepath)
    
def load_ckpoints(weights_path, istrain:bool):
    ckpoints = torch.load(weights_path)
    
    if istrain:
        return ckpoints['model_2'], ckpoints['epoch'], ckpoints['batch_idx'], ckpoints['optimizer_1'], ckpoints['optimizer_2']
    else:
        return ckpoints['model_1']