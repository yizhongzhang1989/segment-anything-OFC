import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class FocalLoss(nn.Module):  
    def __init__(
            self, 
            alpha = 0.75, 
            gamma = 2.0, 
            reduction: Literal['none', 'mean', 'sum'] = 'mean'
        ):
        """
        With inputs as logits.

        alpha:
            If g.t. = 1, then weight = alpha.\n
            If g.t. = 0, then weight = 1 - alpha.
        """
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.reduction = reduction  
  
    def forward(self, inputs, targets):  
        # Cross entropy loss from logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  
          
        # 当targets==1时，pt为sigmoid(inputs)，否则为1-sigmoid(inputs)  
        pt = torch.exp(-BCE_loss)  

        # alpha_t = alpha (if class = 1), = 1 - alpha (if class = 0)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
          
        # 计算focal loss  
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss  
  
        if self.reduction == 'mean':  
            return torch.mean(F_loss)  
        elif self.reduction == 'sum':  
            return torch.sum(F_loss)  
        else:  
            return F_loss  
        

class DiceLoss(nn.Module):  
    def __init__(self, smooth = 1e-5):  
        """
        With inputs as logits.
        """
        super(DiceLoss, self).__init__()  
        self.smooth = smooth
  
    def forward(self, inputs, targets):
        # Flatten the tensor to form a 1-dim vector
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # input logits -> probabilities
        inputs_flat = torch.sigmoid(inputs_flat)
          
        intersection = (inputs_flat * targets_flat).sum()  
        
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (  
            inputs_flat.sum() + targets_flat.sum() + self.smooth)  
          
        return dice_loss  

