import torch
import torch.nn as nn
from torch.autograd import Variable

# output = (torch.FloatTensor([0,0,0,1]))
# target = (torch.FloatTensor([0,0,0,1]))

# criterion = nn.CrossEntropyLoss() #out softmax, loss
# criterion = nn.BCELoss() # loss
# criterion = nn.BCEWithLogitsLoss()
# loss = criterion(output, target)
# print(loss)

def accuracy_fn(pred, label):
    pred = pred.detach() # B , L , C
    label = label.detach() #B, L
    correct = 0
    total = 0
    pred = torch.argmax(pred, dim=2) # B, L
    print(label!=0)
    correct = (label == pred)*(label != 0)
    return  correct.sum()/(label != 0).sum()


target = torch.as_tensor([[    2, 0   ]])
pred = torch.tensor([ 
    [[100, -100, 999],[ 2100, 100, 0]]
])
print(accuracy_fn(pred, target))
