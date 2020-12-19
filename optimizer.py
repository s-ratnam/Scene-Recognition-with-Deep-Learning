'''
This class contains helper functions which will help get the optimizer
'''

import torch


def get_optimizer(model: torch.nn.Module,
                  config: dict) -> torch.optim.Optimizer:
  '''
  Returns the optimizer initializer according to the config on the model.

  Note: config has a minimum of three entries. Feel free to add more entries if you want.
  But do not change the name of the three existing entries

  Args:
  - model: the model to optimize for
  - config: a dictionary containing parameters for the config
  Returns:
  - optimizer: the optimizer
  '''

  optimizer = None

  optimizer_type = config["optimizer_type"]
  learning_rate = config["lr"]
  weight_decay = config["weight_decay"]

  ############################################################################
  # Student code begin
  ############################################################################
  # test with higher momentum
  # test with higher learning rate
  if config["optimizer_type"] == "sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, dampening=0, nesterov=False)
  # test with Adam optimizer -- winner winner 
  elif config["optimizer_type"] == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
  ############################################################################
  # Student code end
  ############################################################################

  return optimizer
