# Blue Dot Project (BDP)
Study into the connection between Locus Coeruleus (LC)'s role in the generation of pathological anxiety and rumination as well as teh effect it would cause on balancing between exploitation and exploration. To this end, we are creating a Locus Coeruleus rumination system in which we call it as the **B**lue **D**ot (LC in greek means little blue dot) **P**roject (BDP).
- Meeting docs: https://docs.google.com/document/d/1740GxJ5xmIjUbWH8_RjYnuI5KNnZwGkvKQaM6hEzCLc/edit?tab=t.0#heading=h.e9mhf81r5r4b

# Notations
## Relationship Between Inputs, LC-NE System, and Outputs

| **Input**                     | **Effect on LC-NE System**         | **Output Affected**                   | **Mathematical Formulation** |
|--------------------------------|----------------------------------|--------------------------------------|--------------------------------|
| **Sensory Stimuli (x_t)**      | Activates LC                     | Modulates all behaviors              | LC_t = f(W_x * x_t + W_h * LC_{t-1} + b) |
| **Acute Stress (S_t)**         | Temporarily spikes LC             | Increases threat learning, avoidance | S_t = S_0 + δ_s * δ(t - t_0) |
| **Chronic Stress (S_t)**       | Sustained LC activation           | Reduces exploration, increases risk aversion | S_t = S_0 + ρ * ∑ e^(-λ (t - k)) * I_k |
| **LC Activity (LC_t)**         | Drives NE release                 | Increases pupil dilation             | P_t = P_0 + η * LC_t |
| **NE Release (NE_t)**          | Governs behavioral responses       | All behaviors depend on NE           | NE_t = g(W_LC * LC_t) |
| **High NE (NE_t)**             | Faster fear learning              | Increased threat learning            | L_t = L_0 + α * NE_t |
| **High NE (NE_t)**             | Less exploratory behavior         | Increased risk aversion              | E_t = E_0 * e^(-β * NE_t) |
| **High NE (NE_t)**             | More avoidance                    | Increases avoidance score            | A_t = 1 / (1 + e^(-γ (NE_t - θ_A))) |
| **High NE (NE_t)**             | Increased anxiety state           | Anxiety accumulation                 | G_t = ∑ e^(-λ (t - k)) * S_k |

## Adjustable Input Variables and Their Effects

| **Input Variable**            | **Description**                                      | **Effect on LC & Behavior**                          | **Mathematical Representation** |
|------------------------------|--------------------------------------------------|-------------------------------------------------|--------------------------------|
| **Sensory Stimulus (x_t)**   | Represents external stimuli (neutral, novel, or threat-related). | Affects LC activation and NE release. Novel stimuli increase LC response. | x_t ~ N(μ_x, σ_x^2) |
| **Stimulus Novelty (N_t)**   | Novelty-driven stimulus intensity.               | Higher novelty causes increased LC activation.  | N_t = e^(-τ (t - t_novel)) |
| **Threat Stimulus (T_t)**    | Threatening stimulus exposure.                   | Strongly activates LC and enhances fear learning. | T_t = θ_T * x_t (where θ_T is the weight of threat exposure) |
| **Acute Stressor (S_t_acute)** | Single strong stressor event.                     | Temporarily increases LC activity, NE release, and anxiety responses. | S_t_acute = S_0 + δ_s * δ(t - t_0) |
| **Chronic Stressor (S_t_chronic)** | Repeated or prolonged stress over time.           | Causes long-term LC hyperactivity and increased baseline anxiety. | S_t_chronic = S_0 + ρ * ∑ e^(-λ (t - k)) * I_k |
| **Top-Down Control (C_t)**   | Prefrontal cortex (PFC) regulation of LC.        | Stronger control suppresses LC activation and prevents anxiety escalation. | C_t = C_0 * e^(-κ S_t) |
| **Environmental Uncertainty (U_t)** | Variability in environmental conditions.          | High uncertainty increases baseline LC activity. | U_t ~ N(0, σ_U^2) |


# Reference Literatures
- Mechanistic Model of Rumination & Cognition: https://onlinelibrary.wiley.com/doi/full/10.1111/tops.12318
- LC & Anxiety: https://pmc.ncbi.nlm.nih.gov/articles/PMC7479871/pdf/10.1177_2398212820930321.pdf 
- Computational Perspective of LC: https://www.sciencedirect.com/science/article/pii/S2352154624000585 
- Rumination Derails Reinforcement Learning With Possible Implications for Ineffective Behavior: https://pmc.ncbi.nlm.nih.gov/articles/PMC9354806/pdf/nihms-1741796.pdf 
- LC-NE Drive RL: https://www.biorxiv.org/content/10.1101/2022.12.08.519670v1.full 
