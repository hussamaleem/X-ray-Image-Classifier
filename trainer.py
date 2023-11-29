import torch

class Trainer():
    
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 criterion,
                 lr_sched):
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.lr_sched = lr_sched
        
    def training_loop(self,train_data):
        
        train_loss = 0
        total_iterations = 0
        train_corr_preds = 0
        total_samples = 0
        
        
        for i,(data,label) in enumerate(train_data):
            
            self.optimizer.zero_grad()
            data,label = data.to(self.device), label.to(self.device)
            pred= self.model(data)
            loss = self.criterion(pred,label)
            loss.backward()
            self.optimizer.step()
            prediction = torch.argmax(pred.data,1)
            corr_pred = (prediction==label).sum().item()
            train_corr_preds+=corr_pred
            train_loss += loss.detach().item()
            
            #self.lr_sched.step()
            total_iterations += 1
            total_samples += label.size(0)
        
            
        total_loss = train_loss/total_iterations
        accuracy = (train_corr_preds / total_samples) * 100
        
        return total_loss, accuracy
        
    def testing_loop(self, test_data):
        
        self.model.eval()
        with torch.no_grad():
            
            test_loss = 0
            total_iterations = 0
            test_corr_preds = 0
            total_samples = 0
            
            for i, (data,label) in enumerate(test_data):
                data,label = data.to(self.device), label.to(self.device)
                pred = self.model(data)
                loss = self.criterion(pred,label)
                prediction = torch.argmax(pred.data,1)
                corr_pred = (prediction==label).sum().item()
                test_corr_preds+=corr_pred
                test_loss+=loss.detach().item()

                total_iterations += 1
                total_samples += label.size(0)
                
            total_test_loss = test_loss/total_iterations
            accuracy = (test_corr_preds / total_samples) * 100
            
            return total_test_loss, accuracy
        
        
        
        
