from torchvision import transforms,datasets
import utils

def create_dataset(train):
        
        dataset_path = utils.get_path(train=train)
        
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        dataset = datasets.ImageFolder(root=dataset_path,
                                       transform=transform)
        
        
        return dataset