import torch
import torch.nn as nn

class LCNEGadget(nn.Module):
    """Self-learned LC-NE system with better Phasic NE adaptation."""
    
    def __init__(self, hidden_dim):
        super(LCNEGadget, self).__init__()
        self.hidden_dim = hidden_dim

        # Deeper LC transformation for more expressivity
        self.W_LC1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC2 = nn.Linear(hidden_dim, hidden_dim)

        # NE Modulation
        self.tonic_control = nn.Linear(hidden_dim, hidden_dim)
        self.phasic_control = nn.Linear(hidden_dim, hidden_dim)

        # Learnable NE scaling
        self.tonic_gain = nn.Parameter(torch.ones(hidden_dim))  
        self.phasic_gain = nn.Parameter(torch.ones(hidden_dim))  

        # Adaptive suppression factor for Phasic NE
        self.suppression_factor = nn.Parameter(torch.ones(hidden_dim))

        # Gating function (learns how to balance tonic & phasic NE)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

        # Normalization layer (prevents runaway NE values)
        self.norm = nn.LayerNorm(hidden_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state):
        """Infers NE modulation dynamically and normalizes output."""
        
        LC_t = self.tanh(self.W_LC1(hidden_state))
        LC_t = self.tanh(self.W_LC2(LC_t))

        tonic_NE = self.tanh(self.tonic_control(LC_t)) * self.tonic_gain  
        phasic_NE = self.tanh(self.phasic_control(LC_t)) * self.phasic_gain  
        phasic_NE *= self.sigmoid(self.suppression_factor * LC_t)

        gating_factor = self.sigmoid(self.gate(LC_t))  
        NE_t = (1 - gating_factor) * tonic_NE + gating_factor * phasic_NE  
        
        NE_t = self.norm(NE_t)  

        return LC_t, NE_t, tonic_NE, phasic_NE


class FFGadgetController(nn.Module):
    """FF network learns to control LC-NE system for modulation"""
    
    def __init__(self, input_dim, hidden_dim):
        super(FFGadgetController, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lcne_gadget = LCNEGadget(hidden_dim) 

        self.modulation_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # tonic & phasic NE
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, activation=False):
        """Processes input and learns to utilize neuromodulation"""
        hidden_1 = torch.relu(self.fc1(x))
        hidden_2 = torch.relu(self.fc2(hidden_1))

        # neuromodulatory signals
        LC_t, NE_t, tonic_NE, phasic_NE = self.lcne_gadget(hidden_2)

        # FFN Learns How to Use Neuromodulation through residuals
        modulated_input = torch.cat([hidden_2, NE_t], dim=1)
        modulated_hidden = torch.relu(self.modulation_fc(modulated_input))
        output = self.output_layer(modulated_hidden)

        if activation:
            return output, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2

        return output, LC_t, NE_t, tonic_NE, phasic_NE


class MemoryLCGadget(nn.Module):
    """Self-learned LC-NE system with integrated short-term memory."""
    
    def __init__(self, hidden_dim):
        super(MemoryLCGadget, self).__init__()
        self.hidden_dim = hidden_dim

        self.W_LC1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC2 = nn.Linear(hidden_dim, hidden_dim)

        self.tonic_control = nn.Linear(hidden_dim, hidden_dim)
        self.phasic_control = nn.Linear(hidden_dim, hidden_dim)

        # short-term memory mechanism (STM)
        self.memory_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # learnable NE scaling
        self.tonic_gain = nn.Parameter(torch.ones(hidden_dim))  
        self.phasic_gain = nn.Parameter(torch.ones(hidden_dim))  

        # adaptive suppression factor for Phasic NE
        self.suppression_factor = nn.Parameter(torch.ones(hidden_dim))

        # gating function (learns how to balance tonic & phasic NE)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

        # normalization layer (prevents runaway NE values)
        self.norm = nn.LayerNorm(hidden_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state, memory_state):
        """Infers NE modulation dynamically while incorporating memory state."""
        
        LC_t = self.tanh(self.W_LC1(hidden_state))
        LC_t = self.tanh(self.W_LC2(LC_t))

        LC_t = LC_t + torch.matmul(memory_state, self.memory_weight)  # memory influence

        # tonic & phasic NE modulation
        tonic_NE = self.tanh(self.tonic_control(LC_t)) * self.tonic_gain  
        phasic_NE = self.tanh(self.phasic_control(LC_t)) * self.phasic_gain  
        phasic_NE *= self.sigmoid(self.suppression_factor * LC_t)

        # adaptive gating between tonic and phasic NE
        gating_factor = self.sigmoid(self.gate(LC_t))  
        NE_t = (1 - gating_factor) * tonic_NE + gating_factor * phasic_NE  
        
        NE_t = self.norm(NE_t)  

        # simple decay, keep track of past activations
        memory_state = 0.9 * memory_state + 0.1 * LC_t

        return LC_t, NE_t, tonic_NE, phasic_NE, memory_state


class FFGadgetControllerWithMemory(nn.Module):
    """FF network learns to control LC-NE system with memory-based modulation."""
    
    def __init__(self, input_dim, hidden_dim):
        super(FFGadgetControllerWithMemory, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # long-term memory mechanism (GRU)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.lcne_gadget = MemoryLCGadget(hidden_dim) 

        self.modulation_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # tonic & phasic NE
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, memory_state=None, activation=False):
        """Processes input and learns to utilize neuromodulation + memory"""
        
        hidden_1 = torch.relu(self.fc1(x))
        hidden_2 = torch.relu(self.fc2(hidden_1))

        if memory_state is None:
            memory_state = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)  # init memory
        
        memory_output, memory_state = self.gru(hidden_2.unsqueeze(1), memory_state)
        memory_output = memory_output.squeeze(1)

        LC_t, NE_t, tonic_NE, phasic_NE, memory_state = self.lcne_gadget(memory_output, memory_state.squeeze(0))

        # FFN Learns How to Use Neuromodulation
        modulated_input = torch.cat([hidden_2, NE_t], dim=1)
        modulated_hidden = torch.relu(self.modulation_fc(modulated_input))
        output = self.output_layer(modulated_hidden)

        if activation:
            return output, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2

        return output, LC_t, NE_t, tonic_NE, phasic_NE
