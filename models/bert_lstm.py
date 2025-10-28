import torch
import torch.nn as nn

class BERTLSTM(nn.Module):
  def __init__(self, pretrained_model, setting='last-layer'):
    super(Model, self).__init__()
    
    assert setting in ['last-layer', 'last-4-layers'], "Setting must be either 'last-layer' or 'last-4-layers'"
    self.setting = setting
    
    self.pretrained_model = pretrained_model
    for param in self.pretrained_model.parameters():
      param.requires_grad = False
      
    self.bilstm1 = nn.LSTM(input_size=768, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)
    self.bilstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)
    self.classifier = nn.Linear(in_features=64, out_features=4)

  def forward(self, inputs):
      
    if self.setting == 'last-layer':
        encoder_outputs = self.pretrained_model(**inputs)['hidden_states'][12]
    else:
        encoder_outputs = self.pretrained_model(**inputs)['hidden_states']
        l12_outputs = encoder_outputs[12]
        l11_outputs = encoder_outputs[11]
        l10_outputs = encoder_outputs[10]
        l9_outputs = encoder_outputs[9]
        encoder_outputs = torch.cat((l9_outputs, l10_outputs, l11_outputs, l12_outputs), dim=1)																					 
    sequence, (_, _) = self.bilstm1(encoder_outputs)
    _, (h, _) = self.bilstm2(sequence)
    h = h.permute(1,0,2)
    h = h.reshape(h.shape[0], -1)
    logits = self.classifier(h)
    return logits
     