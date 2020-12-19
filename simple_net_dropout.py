import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################

    self.cnn_layers =  nn.Sequential(
      nn.Conv2d(1, 10, kernel_size=5, stride=1),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(10, 20, kernel_size = 5, stride = 1),
      nn.Dropout(0.2),
      nn.ReLU(),
      nn.MaxPool2d(3)
    )
    self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.Linear(100, 15))
    self.loss_criterion = nn.CrossEntropyLoss(size_average = None, reduce = None, reduction = 'sum')

    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ############################################################################
    # Student code begin
    ############################################################################
    x = self.cnn_layers(x)
    x = x.view(-1, 500)
    model_output = self.fc_layers(x)

    ############################################################################
    # Student code end
    ############################################################################

    return model_output
