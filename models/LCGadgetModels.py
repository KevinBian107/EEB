import torch
import torch.nn as nn

class LCNEGadget(nn.Module):
    """Neuro-inspired LC-NE system for neuromodulation"""
    
    def __init__(self, hidden_dim):
        super(LCNEGadget, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_LC = nn.Linear(hidden_dim, hidden_dim)
        self.tonic_control = nn.Linear(hidden_dim, hidden_dim)  # Regulates baseline NE
        self.phasic_control = nn.Linear(hidden_dim, hidden_dim)  # Regulates burst-driven NE
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, hidden_state):
        """Modulates NE release based on phasic & tonic LC activity"""
        
        LC_t = self.tanh(self.W_LC(hidden_state))
        tonic_NE = self.sigmoid(self.tonic_control(LC_t))  # Slow, steady modulation
        phasic_NE = self.sigmoid(self.phasic_control(LC_t))  # Fast, event-driven modulation

        NE_t = tonic_NE + phasic_NE

        return LC_t, NE_t, tonic_NE, phasic_NE


class FFGadgetController(nn.Module):
    """FF network learns to control LC-NE system for modulation"""
    
    def __init__(self, input_dim, hidden_dim):
        super(FFGadgetController, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lcne_gadget = LCNEGadget(hidden_dim) 

        self.modulation_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Includes tonic & phasic NE
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, activation=False):
        """Processes input and learns to utilize neuromodulation"""
        hidden_1 = torch.relu(self.fc1(x))
        hidden_2 = torch.relu(self.fc2(hidden_1))

        # Get neuromodulatory signals
        LC_t, NE_t, tonic_NE, phasic_NE = self.lcne_gadget(hidden_2)

        # FFN Learns How to Use Neuromodulation
        modulated_input = torch.cat([hidden_2, NE_t], dim=1)
        modulated_hidden = torch.relu(self.modulation_fc(modulated_input))
        output = self.output_layer(modulated_hidden)

        if activation:
            return output, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2

        return output, LC_t, NE_t, tonic_NE, phasic_NE