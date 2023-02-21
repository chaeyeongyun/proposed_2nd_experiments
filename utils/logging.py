import wandb
from typing import List

#TODO: 개선/ 추가
class Logger():
    def __init__(self, cfg):
        wandb.init(project=cfg.project_name,
                #    notes="baseline",
                #    tags = ["csp+unet+cutmix"]
                )
        # config setting
        d = {}
        for key in cfg.train.wandb_config:
            d[key] = cfg.train[key]
        
        for i in cfg.train.wandb_metrics:
            if i in ["loss"]:
                wandb.define_metric(i, summary='min')
            if i in ["miou"]:
                wandb.define_metric(i, summary='max')
        
    def logging(self, log_dict, epoch):
        wandb.log(log_dict, step=epoch)
        
    
    # def end(self, summary_dict):
    #     for key, value in summary_dict.items():
    #         wandb.run.summary[key] = value