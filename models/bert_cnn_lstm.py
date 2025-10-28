import torch
import torch.nn as nn

class BERTCNNLSTM(nn.Module):
  def __init__(self, pretrained_model, sequence_length, setting='last-layer'):
    super(BERTCNNLSTM, self).__init__()
    
    assert setting in ['last-layer', 'last-4-layers'], "Setting must be either 'last-layer' or 'last-4-layers'"
    
    self.setting = setting
    self.sequence_length = sequence_length if setting == 'last-layer' else sequence_length * 4
    
    self.pretrained_model = pretrained_model
    for param in self.pretrained_model.parameters():
      param.requires_grad = False
      
    self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2)
    self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3)
    self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4)
    self.bilstm = nn.LSTM(input_size=sequence_length*3-6, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(p=0.2)
    self.classifier = nn.Linear(in_features=64, out_features=4)

  def forward(self, inputs):
    if self.setting == 'last-layer':
        encoder_outputs = self.pretrained_model(**inputs)['hidden_states'][12]
    elif self.setting == 'last-4-layers':
        encoder_outputs = self.pretrained_model(**inputs)['hidden_states']
        l12_outputs = encoder_outputs[12]
        l11_outputs = encoder_outputs[11]
        l10_outputs = encoder_outputs[10]
        l9_outputs = encoder_outputs[9]
        encoder_outputs = torch.cat((l9_outputs, l10_outputs, l11_outputs, l12_outputs), dim=1)
    encoder_outputs = encoder_outputs.permute(0, 2, 1)
    conv1_outputs = self.conv1(encoder_outputs)
    conv2_outputs = self.conv2(encoder_outputs)
    conv3_outputs = self.conv3(encoder_outputs)
    concat_outputs = torch.cat((conv1_outputs, conv2_outputs, conv3_outputs), dim=2)
    _, (h, _) = self.bilstm(concat_outputs)
    h = h.permute(1,0,2)
    h = h.reshape(h.shape[0], -1)
    h = self.dropout(h)
    logits = self.classifier(h)
    return logits