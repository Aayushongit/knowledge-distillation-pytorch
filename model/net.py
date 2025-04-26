import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.ln1 = nn.LayerNorm([out_channels, 32//stride, 32//stride])
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm([out_channels, 32//stride, 32//stride])
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.LayerNorm([out_channels, 32//stride, 32//stride])
            )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EnhancedNet(nn.Module):
    def __init__(self, params):
        super(EnhancedNet, self).__init__()
        self.num_channels = params.num_channels
        self.dropout_rate = params.dropout_rate
        self.enable_pruning = getattr(params, 'enable_pruning', False)
        
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm([self.num_channels, 32, 32])
        
        self.res1 = ResidualBlock(self.num_channels, self.num_channels*2, 2)
        self.res2 = ResidualBlock(self.num_channels*2, self.num_channels*4, 2)
        self.res3 = ResidualBlock(self.num_channels*4, self.num_channels*8, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_channels*8, 10)
        
        self._initialize_weights()
        self.masks = None
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def apply_pruning(self, pruning_rate=0.3):
        if not self.enable_pruning:
            return
            
        self.masks = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                threshold = np.percentile(np.abs(tensor), pruning_rate * 100)
                mask = torch.FloatTensor(np.abs(tensor) > threshold).to(param.device)
                self.masks[name] = mask
                param.data *= mask
                
    def forward(self, x):
        if self.masks is not None and self.enable_pruning:
            for name, param in self.named_parameters():
                if 'weight' in name and name in self.masks:
                    param.data *= self.masks[name]
                    
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.res1(out)
        
        if self.training:
            out = F.dropout2d(out, p=self.dropout_rate/2)
            
        out = self.res2(out)
        
        if self.training:
            out = F.dropout2d(out, p=self.dropout_rate)
            
        out = self.res3(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        
        if self.training:
            out = F.dropout(out, p=self.dropout_rate)
            
        out = self.fc(out)
        return out

def weighted_loss_fn(outputs, labels, weights=None):
    if weights is None:
        return nn.CrossEntropyLoss()(outputs, labels)
    else:
        return nn.CrossEntropyLoss(weight=torch.tensor(weights).to(outputs.device))(outputs, labels)

def focal_loss_fn(outputs, labels, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(outputs, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def top_k_accuracy(outputs, labels, k=5):
    batch_size = labels.size
    top_k_preds = np.argsort(outputs, axis=1)[:, -k:]
    correct = 0
    for i in range(batch_size):
        if labels[i] in top_k_preds[i]:
            correct += 1
    return correct / float(batch_size)

metrics = {
    'accuracy': accuracy,
    'top_5_accuracy': lambda outputs, labels: top_k_accuracy(outputs, labels, k=5),
    'f1_score': lambda outputs, labels: compute_f1_score(np.argmax(outputs, axis=1), labels)
}

def compute_f1_score(predictions, labels, average='macro'):
    unique_labels = np.unique(labels)
    f1_scores = []
    
    for label in unique_labels:
        true_positives = np.sum((predictions == label) & (labels == label))
        false_positives = np.sum((predictions == label) & (labels != label))
        false_negatives = np.sum((predictions != label) & (labels == label))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)
    
    if average == 'macro':
        return np.mean(f1_scores)
    else:
        return f1_scores
