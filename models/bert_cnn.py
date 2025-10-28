import torch
import torch.nn as nn

class BERTCNN(nn.Module):
  def __init__(self, pretrained_model, sequence_length, setting='last-layer'):
    super(BERTCNN, self).__init__()
    
    assert setting in ['last-layer', 'last-4-layers'], "Setting must be either 'last-layer' or 'last-4-layers'"
    
    self.setting = setting
    self.sequence_length = sequence_length if setting == 'last-layer' else sequence_length * 4
        
    self.pretrained_model = pretrained_model
    for param in self.pretrained_model.parameters():
      param.requires_grad = False
      
    self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2)
    self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3)
    self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4)
    self.pool1 = nn.MaxPool1d(kernel_size=self.sequence_length-1)
    self.pool2 = nn.MaxPool1d(kernel_size=self.sequence_length-2)
    self.pool3 = nn.MaxPool1d(kernel_size=self.sequence_length-3)
    self.dropout = nn.Dropout(p=0.3)
    self.classifier = nn.Linear(in_features=300, out_features=4)

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
    encoder_outputs = encoder_outputs.permute(0, 2, 1)
    conv1_outputs = self.conv1(encoder_outputs)
    conv2_outputs = self.conv2(encoder_outputs)
    conv3_outputs = self.conv3(encoder_outputs)
    pool1_outputs = self.pool1(conv1_outputs)
    pool2_outputs = self.pool2(conv2_outputs)
    pool3_outputs = self.pool3(conv3_outputs)
    concat_outputs = torch.cat((pool1_outputs, pool2_outputs, pool3_outputs), dim=1).squeeze()
    concat_outputs = self.dropout(concat_outputs)
    logits = self.classifier(concat_outputs)
    return logits