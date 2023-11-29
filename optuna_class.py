import optuna
import dataloader
import hyperparams
import math
import config
import trainer
import utils
import vision
import torch.optim as optim
import torch.nn as nn
from tqdm import trange




class OptunaOptim():
    
    def __init__(self,
                 time_stamp,
                 device):
        
        self.time_stamp = time_stamp
        self.device = device
        self.net = config.net
        self.custom_scheduler = config.custom_scheduler
        
        self.train_dataset, self.test_dataset=dataloader.get_datasets()
        self.total_iter = len(self.train_dataset)
        
    def get_hyperparams(self,trial):
        
        self.params,self.model_params = hyperparams.Hyperparameters(trial=trial,net=self.net).get_hyperparams()
        
    def create_parameters(self):
        
        self.batch_size = self.params['Batch_Size']
        self.lr = self.params['LR']
        self.n_epochs = config.n_epochs
        self.start_epoch = config.start_epoch
        self.repeat_epoch = config.repeat_epoch
        self.init_function = self.params['Initializer']


        
    def get_dataloaders(self):
        
        
        self.train_loader , self.test_loader=dataloader.get_loaders(self.batch_size, 
                                                                    self.train_dataset, 
                                                                    self.test_dataset)
    
    def get_model(self):
        

        network_class = getattr(vision, self.net + 'Network')


        self.model = network_class(**self.model_params).to(self.device)
        
        
        self.model.weight_initialization(self.init_function) 

    def training_setup(self):
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    betas=(0.9,0.98),
                                    eps=1e-09)
        
        if self.custom_scheduler:
            warm_up_steps = math.ceil(self.total_iter/self.batch_size)*self.n_epochs
        

            self.lr_sched =  utils.TransformerScheduler(self.optimizer, 
                                                        d_model = self.d_model, 
                                                        warmup_steps = int(warm_up_steps*self.warmup_ratio))
        else:
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                                 gamma=0.98)
        self.criterion = nn.CrossEntropyLoss()
        
    def objective(self,trial):
        
        test_acc_list = []
        self.get_hyperparams(trial=trial)
        self.create_parameters()
        self.get_model()
        self.get_dataloaders()
        self.training_setup()
        
        self.train_test_class = trainer.Trainer(model=self.model, 
                                                optimizer=self.optimizer, 
                                                device=self.device, 
                                                criterion=self.criterion,
                                                lr_sched=None)
        
        pbar = trange(self.n_epochs)
        
        for epoch in pbar:
            
            pbar.set_description(f'Epoch {epoch}')
            
            train_loss, train_accuracy = self.train_test_class.training_loop(train_data=self.train_loader)

            pbar.set_postfix(Training_Loss = train_loss)
            self.lr_scheduler.step()
            
            if ((epoch+1) > self.start_epoch) and ((epoch+1)%self.repeat_epoch == 0):
                test_loss, test_accuracy = self.train_test_class.testing_loop(test_data=self.test_loader)
                
                utils.model_saver(epoch, 
                                  trial.number, 
                                  self.time_stamp, 
                                  self.start_epoch, 
                                  self.model, 
                                  test_accuracy, 
                                  test_acc_list)
                
                test_acc_list.append(test_accuracy)
                
                pbar.set_postfix(Train_loss=train_loss, 
                                 Test_loss=test_loss,
                                 Train_Accuracy=train_accuracy,
                                 test_Accuracy=test_accuracy)
            
                trial.report(test_accuracy, epoch)
                if trial.should_prune():
                        
                    raise optuna.exceptions.TrialPruned()
            
        return max(test_acc_list)
    
    
    def run_objective(self):
        
        sampler = optuna.samplers.TPESampler(n_startup_trials=config.n_startup_trials,
                                             constant_liar=True)
        
        self.study = optuna.create_study(direction='maximize',
                                         sampler=sampler)
        self.study.optimize(self.objective, 
                            n_trials=config.n_trials, 
                            gc_after_trial=True)