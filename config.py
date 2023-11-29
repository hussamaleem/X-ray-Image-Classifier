import torch

n_epochs = 40
n_trials = 20
n_startup_trials = 8
device = torch.device('mps')
in_channel = 3
out_dim = 2
start_epoch = 10
repeat_epoch = 5
net = 'CNN'
custom_scheduler = False