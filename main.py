import nn_trainer
import nn_trainer.trainers as nn_t
import numpy

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        self.layer1 = nn.Linear(100, 100)

        super().__init__()

    def forward(self, input):
        
        output=self.layer1(input)

        return output

network = Net()

trainer = nn_t.BasicNnTrainer(None, neural_network=network,verbose=True, dtype = torch.float32)

trainer.train()