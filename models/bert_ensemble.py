import torch
import torch.nn as nn

class BERTENSEMBLE(nn.Module):
  def __init__(self, pretrained_model, sequence_length, setting='last-layer'):
    super(BERTENSEMBLE, self).__init__()
    
    assert setting in ['last-layer', 'last-4-layers'], "Setting must be either 'last-layer' or 'last-4-layers'"
    
    self.setting = setting
    self.sequence_length = sequence_length if setting == 'last-layer' else sequence_length * 4
    
    self.pretrained_model = pretrained_model
    for param in self.pretrained_model.parameters():
      param.requires_grad = False

    self.bilstm1 = nn.LSTM(input_size=768, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)
    self.bilstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)
    self.classifier1 = nn.Linear(in_features=64, out_features=4)

    self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2)
    self.conv2 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3)
    self.conv3 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4)
    self.pool1 = nn.MaxPool1d(kernel_size=self.sequence_length-1)
    self.pool2 = nn.MaxPool1d(kernel_size=self.sequence_length-2)
    self.pool3 = nn.MaxPool1d(kernel_size=self.sequence_length-3)
    self.dropout = nn.Dropout(p=0.3)
    self.classifier2 = nn.Linear(in_features=300, out_features=4)

  def forward(self, inputs):
    if self.setting == 'last-layer':
        encoder_outputs = self.pretrained_model(**inputs)['hidden_states'][12]               # (B, S, 768)
    else:
        encoder_outputs = self.pretrained_model(**inputs)['hidden_states']
        l12_outputs = encoder_outputs[12]
        l11_outputs = encoder_outputs[11]
        l10_outputs = encoder_outputs[10]
        l9_outputs = encoder_outputs[9]
        encoder_outputs = torch.cat((l9_outputs, l10_outputs, l11_outputs, l12_outputs), dim=1)            #   (B, S*4, 768)
        
    # ----------------------BiLSTM------------------
    sequence, (_, _) = self.bilstm1(encoder_outputs)                                  # (B, S, 64) or  (B, S*4, 64)
    _, (h, _) = self.bilstm2(sequence)                                                # (2, B, 32)
    h = h.permute(1,0,2)                                                              # (B, 2, 32)
    h = h.reshape(h.shape[0], -1)                                                     # (B, 64)
    logits1 = self.classifier1(h)                                                     # (B, 4)
    # ----------------------------------------------
    
    #----------------------CNN----------------------
    encoder_outputs = encoder_outputs.permute(0, 2, 1)                                # (B, 768, S) or   (B, 768, S*4)
    conv1_outputs = self.conv1(encoder_outputs)                                       # (B, 100, S-1) or (B, 100, S*4-1)
    conv2_outputs = self.conv2(encoder_outputs)                                       # (B, 100, S-2) or (B, 100, S*4-2)
    conv3_outputs = self.conv3(encoder_outputs)                                       # (B, 100, S-3) or (B, 100, S*4-3)
    pool1_outputs = self.pool1(conv1_outputs)                                         # (B, 100, 1)
    pool2_outputs = self.pool2(conv2_outputs)                                         # (B, 100, 1)
    pool3_outputs = self.pool3(conv3_outputs)                                         # (B, 100, 1)
    concat_outputs = torch.cat((pool1_outputs, pool2_outputs, pool3_outputs), dim=1).squeeze() # (B, 300)
    concat_outputs = self.dropout(concat_outputs)                                     # (B, 300)
    logits2 = self.classifier2(concat_outputs)                                        # (B, 4)
    # ----------------------------------------------

    return torch.add(logits1, logits2)