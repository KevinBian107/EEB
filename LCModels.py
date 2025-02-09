import torch
import torch.nn as nn
import numpy as np

class LCNECortexFCN(nn.Module):
    '''Vanilla FCN based LCNE model'''
    def __init__(self, lambda_cortex):
        super(LCNECortexFCN, self).__init__()
        
        self.lambda_cortex = lambda_cortex
        self.W_x = nn.Parameter(torch.tensor(0.5))  # Input weight to LC
        self.W_h = nn.Parameter(torch.tensor(0.7))  # LC recurrent weight
        self.W_LC = nn.Parameter(torch.tensor(0.8))  # LC-to-Cortex modulation
        self.W_C = nn.Parameter(torch.tensor(0.5))  # Stimulus-to-Cortex weight
        self.b = nn.Parameter(torch.tensor(0.1))  # LC bias

    def forward(self, x_t, stress_t, prev_LC, prev_Cortex):
        # LC Activity Update
        LC_t = torch.tanh(self.W_x * x_t + self.W_h * prev_LC + self.b + stress_t)
        
        # NE Release
        NE_t = torch.sigmoid(self.W_LC * LC_t)
        
        # Cortical Activity Update
        C_t = prev_Cortex + self.lambda_cortex * self.W_LC * NE_t + self.W_C * x_t
        
        return LC_t, NE_t, C_t

class LCNECortexRNN(nn.Module):
    '''An RNN based LCNE model'''
    def __init__(self, lambda_cortex, hidden_size=16):
        super(LCNECortexRNN, self).__init__()
        self.lambda_cortex = lambda_cortex
        self.hidden_size = hidden_size
        
        self.lc_rnn = nn.RNNCell(input_size=hidden_size, hidden_size=hidden_size)
        self.cortex_rnn = nn.RNNCell(input_size=hidden_size, hidden_size=hidden_size)

        self.lc_out = nn.Linear(hidden_size, 1)  # LC output
        self.ne_out = nn.Linear(hidden_size, 1)  # NE output
        self.cortex_out = nn.Linear(hidden_size, 1)  # Cortex output

    def forward(self, x_t, stress_t, prev_LC_hidden, prev_Cortex_hidden):
        # Expand x_t to match hidden size
        x_t_exp = x_t.expand(1, self.hidden_size)  # Expand to (batch_size, hidden_size)
        
        # Update LC hidden state
        LC_hidden = self.lc_rnn(x_t_exp, prev_LC_hidden)
        
        # Compute LC output and NE release
        LC_t = torch.tanh(self.lc_out(LC_hidden))
        NE_t = torch.sigmoid(self.ne_out(LC_hidden))

        # Update Cortex hidden state
        NE_exp = NE_t.expand(1, self.hidden_size)  # Ensure correct input size
        Cortex_hidden = self.cortex_rnn(NE_exp, prev_Cortex_hidden)

        # Compute Cortex output
        C_t = self.cortex_out(Cortex_hidden) + self.lambda_cortex * NE_t

        return LC_t, NE_t, C_t, LC_hidden, Cortex_hidden

class LCNECortexFitter(nn.Module):
    def __init__(self, lambda_cortex=0.1):
        super(LCNECortexFitter, self).__init__()
        self.lambda_cortex = lambda_cortex

        # learnable parameters to optimized for fiting real data
        self.W_x = nn.Parameter(torch.tensor(0.5))  # Sensory input weight
        self.W_h = nn.Parameter(torch.tensor(0.7))  # LC recurrent weight
        self.W_LC = nn.Parameter(torch.tensor(0.8))  # LC-to-Cortex modulation
        self.W_C = nn.Parameter(torch.tensor(0.5))  # Sensory-to-Cortex weight
        self.b = nn.Parameter(torch.tensor(0.1))  # LC bias

    def forward(self, X, prev_LC, prev_Cortex):
        """
        X: Input features (batch_size, num_features)
        prev_LC: Previous LC activation (batch_size,)
        prev_Cortex: Previous Cortex activation (batch_size,)
        """
        # Compute LC activity update for batch
        LC_t = torch.tanh(self.W_x * X[:, 0] + self.W_h * prev_LC + self.b)

        # Compute NE release
        NE_t = torch.sigmoid(self.W_LC * LC_t)

        # Compute cortical activity update
        C_t = prev_Cortex + self.lambda_cortex * self.W_LC * NE_t + self.W_C * X[:, 0]

        # Predicted pupil dilation (arousal effect)
        Pupil_t = 0.4 * LC_t + 0.3 * C_t + 0.2 * NE_t + 0.1 * torch.sin(X[:, 0])

        return LC_t, NE_t, C_t, Pupil_t

class PupilDilationPredictor(nn.Module):
    '''Simple linear regression model to predict pupil dilation'''
    def __init__(self, input_size):
        super(PupilDilationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x