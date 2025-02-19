import torch
import torch.nn as nn

class LCNEGadget(nn.Module):
    """A differentiable LC-NE inspired neuromodulation unit"""
    def __init__(self, hidden_dim):
        super(LCNEGadget, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LC Dynamics
        self.W_LC = nn.Linear(hidden_dim, hidden_dim)
        self.W_NE = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating Mechanisms
        self.forget_gate = nn.Linear(hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, hidden_state):
        """Takes in hidden state and produces LC-NE modulated control signals"""
        
        LC_t = self.tanh(self.W_LC(hidden_state))  # LC activation
        NE_t = self.sigmoid(self.W_NE(LC_t))  # NE neuromodulation
        
        forget_signal = self.sigmoid(self.forget_gate(hidden_state))  # Forget previous state
        input_signal = self.sigmoid(self.input_gate(hidden_state))  # Influence of new input
        output_signal = self.sigmoid(self.output_gate(hidden_state))  # Output modulation
        
        return LC_t, NE_t, forget_signal, input_signal, output_signal


class LSTMGadget(nn.Module):
    """LSTM that learns to use an LC-NE neuromodulation gadget"""
    def __init__(self, input_dim, hidden_dim):
        super(LSTMGadget, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lcne_gadget = LCNEGadget(hidden_dim)  # Our neuromodulation unit

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_0 = torch.zeros(1, batch_size, self.hidden_dim)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # Apply LC-NE gadget to modulate LSTM hidden states
        LC_t, NE_t, forget_signal, input_signal, output_signal = self.lcne_gadget(h_n.squeeze(0))

        # Modulate hidden states based on LC-NE activity
        modulated_hidden = forget_signal * h_n.squeeze(0) + input_signal * NE_t
        
        # Compute final output
        output = self.output_layer(modulated_hidden)

        return output, LC_t, NE_t, forget_signal, input_signal, output_signal
    

class LCNEGadget(nn.Module):
    """A neuro-inspired LC-NE gadget that produces modulating signals"""
    def __init__(self, hidden_dim):
        super(LCNEGadget, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LC Dynamics
        self.W_LC = nn.Linear(hidden_dim, hidden_dim)
        self.W_NE = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating Mechanisms
        self.forget_gate = nn.Linear(hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, hidden_state):
        """Takes in hidden state and produces LC-NE modulated control signals"""
        
        LC_t = self.tanh(self.W_LC(hidden_state))  # LC activation
        NE_t = self.sigmoid(self.W_NE(LC_t))  # NE neuromodulation
        
        forget_signal = self.sigmoid(self.forget_gate(hidden_state))  # Forget previous state
        input_signal = self.sigmoid(self.input_gate(hidden_state))  # Influence of new input
        output_signal = self.sigmoid(self.output_gate(hidden_state))  # Output modulation
        
        return LC_t, NE_t, forget_signal, input_signal, output_signal


class FFControllerWithLCNEGadget(nn.Module):
    """Feedforward NN that learns to control an LC-NE neuromodulation gadget"""
    def __init__(self, input_dim, hidden_dim):
        super(FFControllerWithLCNEGadget, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lcne_gadget = LCNEGadget(hidden_dim)

        self.modulation_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Combine FF + LC-NE signals

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, activation=False):
        """Processes input with FFN and learns to use LC-NE gadget"""
        hidden_1 = torch.relu(self.fc1(x))
        hidden_2 = torch.relu(self.fc2(hidden_1))

        LC_t, NE_t, forget_signal, input_signal, output_signal = self.lcne_gadget(hidden_2)

        # FFN Learns How to Use Neuromodulation
        modulated_input = torch.cat([hidden_2, NE_t], dim=1)
        modulated_hidden = self.modulation_fc(modulated_input)  # FF decides usage
        
        output = self.output_layer(modulated_hidden)
        
        if activation:
            return output, LC_t, NE_t, forget_signal, input_signal, output_signal, hidden_1, hidden_2
        
        return output, LC_t, NE_t, forget_signal, input_signal, output_signal
