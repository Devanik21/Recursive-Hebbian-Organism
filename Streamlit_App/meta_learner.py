"""
MetaLearner: The "God Optimizer" for Differentiable Plasticity.

This module implements the Outer Loop of meta-learning. It evolves the 
PlasticityNetwork (Genome) by:
1. Running the brain through a "lifetime" of learning (Inner Loop)
2. Measuring task performance (Outer Loop Loss)
3. Backpropagating through time to update the Genome parameters

Key Safety Features:
- Device Agnosticism: Tensors automatically move to brain's device (CPU/GPU)
- State Decontamination: Memory buffers cleared between episodes
- Functional Weights: Gradient graph preserved for backpropagation
- Weight Normalization: Prevents explosion during learning

References:
- Uber AI: "Differentiable Plasticity" (2018)
- Teacher's "Brutal Evaluation" refinements
"""

import torch
import torch.nn.functional as F


class MetaLearner:
    """
    The "God Optimizer": Evolves the plasticity rule through meta-learning.
    
    Architecture:
        Inner Loop: Brain learns using the Genome's plasticity rule (no backprop)
        Outer Loop: Task loss backpropagates to update the Genome
    
    This creates a brain that "learns to learn" - it starts with random weights
    but has an evolved, optimal learning rule.
    """
    
    def __init__(self, brain, genome, lr: float = 0.001, plasticity_lr: float = 0.1):
        """
        Initialize the MetaLearner.
        
        Args:
            brain: The PlasticCortex instance
            genome: The PlasticityNetwork instance (learnable plasticity rule)
            lr: Learning rate for the Genome optimizer (outer loop)
            plasticity_lr: Scaling factor for weight updates (inner loop)
        """
        self.brain = brain
        self.genome = genome
        self.plasticity_lr = plasticity_lr
        
        # The "God Optimizer" - updates the Genome's parameters
        self.optimizer = torch.optim.Adam(genome.parameters(), lr=lr)
        
        # === ASSOCIATIVE TASK GENERATOR (Fixed Projection) ===
        # This creates a DETERMINISTIC mapping from cue -> target.
        # The projection is FROZEN (not learned) to provide stable ground truth.
        input_dim = brain.synapse.shape[0]   # Embedding dimension (e.g., 32)
        hidden_dim = brain.synapse.shape[1]  # Target dimension (e.g., 1024)
        self.target_projection = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        # Freeze it - this is NOT learned, just a fixed function
        for param in self.target_projection.parameters():
            param.requires_grad = False
        
        # Tracking metrics
        self.loss_history = []
        self.episode_count = 0
    
    def meta_step(self, num_inner_steps: int = 5, seq_length: int = 10) -> float:
        """
        Execute one meta-learning episode.
        
        This runs:
        1. Pre-flight safety checks (device, memory reset)
        2. Inner Loop: Lifetime learning with functional weights
        3. Outer Loop: Task loss calculation and Genome evolution
        
        Args:
            num_inner_steps: Number of inner loop iterations (keep low to avoid OOM)
            seq_length: Length of synthetic byte sequences
            
        Returns:
            loss: The task loss for this episode (lower = better learning rule)
        """
        # === 0. PRE-FLIGHT SAFETY CHECKS ===
        
        # Fix: Device Agnosticism (Avoid CPU/GPU mismatch crash)
        device = self.brain.synapse.device
        
        # Move target projection to correct device
        self.target_projection = self.target_projection.to(device)
        
        # Fix: State Decontamination (Prevent memory bleeding between episodes)
        with torch.no_grad():
            self.brain.short_term_latent.fill_(0)
            self.brain.long_term_latent.fill_(0)
        
        # === 1. CREATE SYNTHETIC ASSOCIATION TASK (Cue -> Target) ===
        
        # Fix: Use randint (Long) because embeddings expect integers, not floats!
        cue = torch.randint(0, 256, (1, seq_length)).long().to(device)
        
        # Target: A DETERMINISTIC function of the cue (Associative Task Generator)
        # This solves the "Invisible Target" paradox - the target is now STABLE.
        with torch.no_grad():
            cue_embedding = self.brain.byte_embed(cue).mean(dim=1)  # [1, input_dim]
        # Project to target space using the FIXED projection
        target_signal = self.target_projection(cue_embedding)  # [1, hidden_dim]
        target_signal = torch.tanh(target_signal)  # Normalize to [-1, 1] range
        
        # === 2. INITIALIZE FAST WEIGHTS (Functional approach) ===
        # Clone to create a copy that's part of the computation graph
        fast_weights = self.brain.synapse.clone()
        
        # === 3. INNER LOOP: LIFETIME LEARNING ===
        # The brain sees the data and updates 'fast_weights' using the Genome
        
        for step in range(num_inner_steps):
            # Forward pass using CURRENT fast_weights
            # Fix: Capture 'pre' (signal) from the updated return signature
            activation, _, pre = self.brain(cue, override_weights=fast_weights)
            
            # CRITICAL: Clamp activation towards target so Hebbian can learn
            # This teaches the rule: "When I see this pattern, activate like this"
            post = 0.5 * activation + 0.5 * target_signal
            
            # Ask Genome for the weight update
            delta_w = self.genome(pre, post, fast_weights)
            
            # Functional Update (Preserves Gradient Graph!)
            # Teacher's "Diamond Polish": Normalize to prevent explosion
            updated_weights = fast_weights + self.plasticity_lr * delta_w
            fast_weights = F.normalize(updated_weights, p=2, dim=1)
        
        # === 4. OUTER LOOP: TEST (Recall) ===
        # Present Cue ONLY, check if the brain recalls the Target
        final_activation, _, _ = self.brain(cue, override_weights=fast_weights)
        
        # Loss: How close was the recall to the target?
        loss = F.mse_loss(final_activation, target_signal)
        
        # === 5. EVOLVE THE GENOME ===
        self.optimizer.zero_grad()
        loss.backward()  # Backprops through time -> fast_weights -> delta_w -> genome
        self.optimizer.step()
        
        # Track metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.episode_count += 1
        
        return loss_value
    
    def train(self, num_episodes: int = 100, verbose: bool = True) -> list:
        """
        Run multiple meta-learning episodes.
        
        Args:
            num_episodes: Number of meta-learning episodes
            verbose: Whether to print progress
            
        Returns:
            losses: List of loss values for each episode
        """
        losses = []
        
        for episode in range(num_episodes):
            loss = self.meta_step()
            losses.append(loss)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_loss = sum(losses[-10:]) / 10
                print(f"Episode {episode + 1}/{num_episodes} | Loss: {loss:.4f} | Avg(10): {avg_loss:.4f}")
        
        return losses
    
    def get_evolved_genome(self):
        """Return the evolved Genome for deployment."""
        return self.genome
    
    def get_loss_history(self):
        """Return the full loss history for visualization."""
        return self.loss_history


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing MetaLearner (The God Optimizer)...")
    
    # Import dependencies
    from core import PlasticCortex
    from plasticity_network import PlasticityNetwork
    
    # Create components
    brain = PlasticCortex(input_dim=32, hidden_dim=64)
    genome = PlasticityNetwork(hidden_dim=16)
    
    print(f"   Brain synapse shape: {brain.synapse.shape}")
    print(f"   Genome parameters: {sum(p.numel() for p in genome.parameters())}")
    
    # Create MetaLearner
    meta_learner = MetaLearner(brain, genome, lr=0.001, plasticity_lr=0.1)
    
    # Run a few meta-steps
    print("\n   Running 5 meta-learning episodes...")
    for i in range(5):
        loss = meta_learner.meta_step(num_inner_steps=3)
        print(f"   Episode {i+1}: Loss = {loss:.4f}")
    
    # Check gradient flow
    print("\n   Verifying gradient flow to Genome...")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in genome.parameters())
    print(f"   Genome received gradients: {has_grad}")
    
    if has_grad:
        print("[PASS] MetaLearner test PASSED!")
    else:
        print("[FAIL] No gradients reached the Genome - check gradient flow!")
