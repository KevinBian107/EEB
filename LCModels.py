import torch
import torch.nn as nn

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
        LC_t = torch.tanh(self.W_x * x_t + self.W_h * prev_LC + self.b + stress_t)
        
        NE_t = torch.sigmoid(self.W_LC * LC_t)
        
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
        x_t_exp = x_t.expand(1, self.hidden_size)  # Expand to (batch_size, hidden_size)
        
        LC_hidden = self.lc_rnn(x_t_exp, prev_LC_hidden)
        
        LC_t = torch.tanh(self.lc_out(LC_hidden))
        NE_t = torch.sigmoid(self.ne_out(LC_hidden))

        NE_exp = NE_t.expand(1, self.hidden_size)  # Ensure correct input size
        Cortex_hidden = self.cortex_rnn(NE_exp, prev_Cortex_hidden)

        C_t = self.cortex_out(Cortex_hidden) + self.lambda_cortex * NE_t

        return LC_t, NE_t, C_t, LC_hidden, Cortex_hidden

class LCNECortexFitterBasic(nn.Module):
    '''Basic LCNECortex model with learnable parameters'''
    def __init__(self, lambda_cortex=0.1):
        super(LCNECortexFitter, self).__init__()
        self.lambda_cortex = lambda_cortex

        # learnable parameters to optimized for fiting real data, initialize with some values
        self.W_x = nn.Parameter(torch.tensor(0.5))  # Sensory input weight
        self.W_h = nn.Parameter(torch.tensor(0.7))  # LC recurrent weight
        self.W_LC = nn.Parameter(torch.tensor(0.8))  # LC-to-Cortex modulation
        self.W_C = nn.Parameter(torch.tensor(0.5))  # Sensory-to-Cortex weight
        self.W_LC_Pupil = nn.Parameter(torch.tensor(0.4))
        self.W_C_Pupil = nn.Parameter(torch.tensor(0.3))
        self.W_NE_Pupil = nn.Parameter(torch.tensor(0.2))
        self.b = nn.Parameter(torch.tensor(0.1))  # LC bias

    def forward(self, X, prev_LC, prev_Cortex):
        #  LC activity update for batch
        LC_t = torch.tanh(self.W_x * X[:, 0] + self.W_h * prev_LC + self.b)

        # NE release
        NE_t = torch.sigmoid(self.W_LC * LC_t)

        # cortical activity update
        C_t = prev_Cortex + self.lambda_cortex * self.W_LC * NE_t + self.W_C * X[:, 0]

        # pupil dilation (arousal effect)
        # Pupil_t = 0.4 * LC_t + 0.3 * C_t + 0.2 * NE_t + 0.1 * torch.sin(X[:, 0])
        Pupil_t = self.W_LC_Pupil * LC_t + self.W_C_Pupil * C_t + self.W_NE_Pupil * NE_t  # Learnable weighting

        return LC_t, NE_t, C_t, Pupil_t
    

class LCNECortexFitter(nn.Module):
    """Improved LCNECortex model with learnable parameters and batch normalization"""
    def __init__(self, input_dim, hidden_dim=8, lambda_cortex=0.1):
        super(LCNECortexFitter, self).__init__()
        self.lambda_cortex = lambda_cortex
        self.hidden_dim = hidden_dim

        # Linear layers to correctly map dimensions
        self.W_x = nn.Linear(input_dim, hidden_dim)  # Now takes full input feature size
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC = nn.Linear(hidden_dim, hidden_dim)
        self.W_C = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC_Pupil = nn.Linear(hidden_dim, 1)  # Output layer for pupil dilation

        # Batch normalization layers for stable training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.LeakyReLU(0.1)  # Prevents dead neurons

    def forward(self, X, prev_LC, prev_Cortex, return_activations=False):
        x_input = self.W_x(X)

        # Compute LC activity
        LC_raw = x_input + self.W_h(prev_LC)  
        LC_t = self.relu(self.bn1(LC_raw))

        # Compute NE release
        NE_raw = self.W_LC(LC_t)
        NE_t = self.relu(self.bn2(NE_raw))

        # Compute cortical activity update
        C_raw = prev_Cortex + self.lambda_cortex * self.W_LC(NE_t) + self.W_C(x_input)
        C_t = self.relu(self.bn3(C_raw))

        # Compute pupil dilation
        Pupil_t = self.W_LC_Pupil(LC_t) + self.W_LC_Pupil(C_t) + self.W_LC_Pupil(NE_t)

        if return_activations:
            return LC_t, NE_t, C_t, Pupil_t, LC_raw, NE_raw, C_raw
        return LC_t, NE_t, C_t, Pupil_t

class FeedForwardNN(nn.Module):
    '''Simple feedforward network with intermediate activation extraction'''
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, return_activations=False):
        act1 = torch.relu(self.fc1(x))
        act2 = torch.relu(self.fc2(act1))
        output = self.fc3(act2)

        if return_activations:
            return output, act1, act2
        return output

class RecurrentNet(nn.Module):  # Ensure nn.Module is inherited
    '''An RNN-based model for predicting pupil dilation'''
    def __init__(self, input_size, hidden_size=32):
        super().__init__()  # Use `super()` correctly without explicit arguments
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # batch_first=True ensures (batch, seq, feature)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)  # Initialize hidden state
        out, _ = self.rnn(x, h0)  # x shape: (batch_size, seq_length=1, input_size)
        out = self.fc(out[:, -1, :])  # Take last timestep output
        return out

