import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size = (224,224), 
                 patch_size =(16,16), in_channels =3, 
                 embed_dim = 64):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embeddings = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        patches = self.patch_embeddings(x)  # Extract patches from the input image
        batch_size, channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)  # Reshape patches
        return patches

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_patches, num_heads):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                                                    nhead=num_heads,
                                                                                    dim_feedforward=2*embed_dim,
                                                                                    batch_first=True),num_layers=4)
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(28),
            nn.Flatten(),
            nn.Linear(28*embed_dim, embed_dim))
        
        
    def forward(self, x):
        patches = self.patch_embedding(x)
        embeddings = patches + self.positional_encoding[:, :patches.size(1)]
        encoded = self.transformer_encoder(embeddings)
        encoded = encoded.permute(0,2,1)
        projection = self.projection(encoded)
        
        return projection
    
class MLPDecoder(nn.Module):
    def __init__(self, out_feature, embed_dim):
        super(MLPDecoder, self).__init__()
    
        self.mlpdecoder = nn.Sequential(
                        nn.Linear(embed_dim,out_feature),
                        nn.LeakyReLU(),
                        nn.Linear(out_feature,out_feature//2),
                        nn.LeakyReLU(),
                        nn.Linear(out_feature//2, 2))
    
    def forward(self,features):
        
        output = self.mlpdecoder(features)
    
        return output
    
class TransformerNetwork(nn.Module):
    
    def __init__(self, image_size,patch_size, in_channels, embed_dim, num_patches,
                 num_heads, out_feature):
        super(TransformerNetwork, self).__init__()
        self.encoder = VisionTransformer(image_size=image_size, 
                                         patch_size = patch_size, 
                                         in_channels = in_channels, 
                                         embed_dim = embed_dim, 
                                         num_patches = num_patches, 
                                         num_heads = num_heads)
        
        self.decoder = MLPDecoder(out_feature=out_feature, 
                                  embed_dim=embed_dim)
        
    def forward(self,image):
        
        encoder_out = self.encoder(image)
        decoder_out = self.decoder(encoder_out)
        
        return decoder_out
    
class CNNNetwork(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_features,
                 out_dim):
        
        super(CNNNetwork, self).__init__()

        self.cnn = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5,5)),
                    nn.MaxPool2d((2,2)),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=out_channels, out_channels=(out_channels//2), kernel_size=(5,5)),
                    nn.MaxPool2d((2,2)),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=(out_channels//2), out_channels=((out_channels)//4), kernel_size=(3,3)),
                    nn.MaxPool2d((2,2)),
                    nn.Flatten())
        
        in_features = (out_channels//4)*25*25
        
        self.fc_1 = nn.Linear(in_features, out_features)
        self.fc_2 = nn.Linear(out_features, out_dim)
        
    def forward(self,src):
        
        out_1 = self.cnn(src)
        out_2 = self.fc_1(out_1)
        out = self.fc_2(out_2)
        
        return out
    
    def weight_initialization(self,init_function):
        if init_function == 'xavier_n':
            initializer = nn.init.xavier_normal_
        elif init_function == 'kaiming_n':
            initializer = nn.init.kaiming_normal_
        elif init_function == 'xavier_u':
            initializer = nn.init.xavier_uniform_
        elif init_function == 'kaiming_u':
            initializer = nn.init.kaiming_uniform_
        for m in self.modules():
            if type(m) is nn.Conv2d:
                for name, param in m.named_parameters():
                    if 'weight' in name:       
                        initializer(param.data)
            if type(m) is nn.Linear:
                for name, param in m.named_parameters():
                    if 'weight' in name:                        
                        initializer(param.data) 
                        
                        
    