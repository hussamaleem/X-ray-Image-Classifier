import os
import torch

def get_path(train):
    
    if train:
        method = 'train'
    else:
        method = 'test'
    dataset_path = os.path.join(os.getcwd(), 'chest_xray', 
                                method)
    
    return dataset_path
    
    
def model_saver(epoch,n_trial,time_stamp,start_epoch,model,score_val,score_list):
        
        sav_path = os.path.join(os.getcwd(),'models', time_stamp , 'Num_trial ' + str(n_trial))
        if not os.path.exists(sav_path):
                    os.makedirs(sav_path)
        
        
        if epoch == (start_epoch+9):
            
            torch.save(model.state_dict(), sav_path +  '/Model Parameters')
            torch.save(model, sav_path + '/Model.pt')
            
        elif (epoch > (start_epoch+9)) and (score_val < min(score_list)):
            
            torch.save(model.state_dict(),sav_path + '/Model Parameters')

            torch.save(model, sav_path + '/Model.pt')
            
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def get_learning_rate(self):
        factor = 0.5 * self.d_model ** (-0.5)
        
        
        if self.step_num < self.warmup_steps:
            scale = 0.25 * (self.step_num*self.warmup_steps**(-1.5))
        else:
            scale = 0.25 * (self.warmup_steps*self.warmup_steps**(-1.5))

        return factor * scale