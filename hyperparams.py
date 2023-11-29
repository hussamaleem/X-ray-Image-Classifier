import config

class Hyperparameters():
    
    def __init__(self,trial,net):
        
        self.trial = trial
        self.net = net
        
    def LR_scheduler(self):
        
        if 'CNN' in self.net:

            lr = self.trial.suggest_float("Learning_Rate",
                                      low=0.001,
                                      high=0.01,
                                      log=True)
    
            warmup_ratio = 0
            
            scheduler_params = {'Warm_Up_Ratio':warmup_ratio,
                                'LR': lr}
        
        else:
            warmup_ratio = self.trial.suggest_float("LR_Warmup_Ratio", 
                                                    low = 0.4, 
                                                    high = 0.75, 
                                                    step = 0.05)
            lr = 0
        
            scheduler_params = {'Warm_Up_Ratio': warmup_ratio,
                                'LR': lr}
        
        return scheduler_params
    
    def train_params(self):
        
        self.out_feature  = self.trial.suggest_int('Out_Feature', 
                                                   low = 48,
                                                   high = 160, 
                                                   step = 16)
        
        batch_size  = self.trial.suggest_int('Batch_Size', 
                                              low = 8,
                                              high = 32, 
                                              step = 4)
        
        initializer = self.trial.suggest_categorical('Initializer',
                                                     ['xavier_u',
                                                      'kaiming_u',
                                                      'xavier_n',
                                                      'kaiming_n'])
        
        training_params = {
                           'Batch_Size': batch_size,
                           'Initializer': initializer}
        
        return training_params
    
    def CNN_params(self):
        
        out_channel =  self.trial.suggest_int('Out_Channel', 
                                              low = 32,
                                              high = 64, 
                                              step = 8)
        
        cnn_params = {
                    'in_channels': config.in_channel,
                    'out_channels': out_channel,
                    'out_features': self.out_feature,
                    'out_dim': config.out_dim
                    }
        
        return cnn_params
        
    
    def transformer_params(self):
        
        d_model  = self.trial.suggest_int('d_model', 
                                          low = 512,
                                          high = 1024, 
                                          step = 16)
        
        num_head = self.trial.suggest_categorical('num_head',
                                                  [8,16])
        
        trans_params = {
                            'image_size': (224, 224),
                            'patch_size': (16, 16),
                            'in_channels': config.in_channel,
                            'embed_dim': d_model,
                            'num_patches': 196,
                            'num_heads': num_head,
                            'out_feature': self.out_feature
                            }
        
        return trans_params
        
    
    def get_hyperparams(self):
        
        scheduler_params = self.LR_scheduler()
        training_params = self.train_params()
        model_params = self.transformer_params() if 'Transformer' in self.net else self.CNN_params()
        
        return {**training_params, **scheduler_params},model_params
    
    
    