import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features 

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        # Prepare features for input
        features = features.unsqueeze(1)
        
        # Prepare captions for input
        captions = captions[:, :-1]
        captions = self.embed(captions)
        
        # Combined features and captions by concatenating
        inputs = torch.cat((features, captions), 1)
        
        # Pass through network
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        "Accepts pre-processed image tensor and returns predicted sentence list"
        sentence = []
        
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.fc(outputs.squeeze(1))
            target_i = outputs.max(1)[1]
            sentence.append(target_i.item())
            inputs = self.embed(target_i).unsqueeze(1)
            
        return sentence
