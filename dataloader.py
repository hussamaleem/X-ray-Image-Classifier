from torch.utils.data import DataLoader
import dataset

def get_datasets():
    
    train_dataset = dataset.create_dataset(train=True)
    test_dataset = dataset.create_dataset(train=False)
    
    return train_dataset, test_dataset

def get_loaders(batch_size,
                train_dataset,
                test_dataset):
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_loader,test_loader
