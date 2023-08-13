



import torch
import torch.nn as nn


model_name_da = "aubmindlab/bert-base-arabertv02-twitter"
model_name = model_name_da
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from torch.autograd import Function


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainAdaptationModel(nn.Module):
    def __init__(self,embed_size= 512, hidden_size = 256, dropout=0.1):
        super(DomainAdaptationModel, self).__init__()

   

        self.dropout = nn.Dropout(dropout)
       
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(dim=1),
        )


    def forward(
          self,
         pooled_output,
          grl_lambda = 1.0,
          ):



        pooled_output = self.dropout(pooled_output)


        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)

    
        domain_pred = self.domain_classifier(reversed_pooled_output)

        return domain_pred.to(device)

class Disc_model(nn.Module):
    def __init__(self, hidden_size=256, input_size=512):
        super(Disc_model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, enc_outputs, label ="trgt"):
     
        h = self.tanh(self.linear(enc_outputs))#(N,hid_dim)
        z = self.fc(h) #(N,2)

        prob = self.logsoftmax(z)

        
        return prob
    