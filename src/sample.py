import torch

class Sampler:
    def __init__(self):
        """Initialize the Sampler."""
        pass
    
    def sample(self, logits):
        """Sample from the logits. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

class GreedySampler(Sampler):
    def __init__(self):
        super().__init__()  # Call the parent constructor
    
    def sample(self, logits):
        """Sample from the logits."""
        return torch.argmax(logits, dim=-1)


class SynthIdSampler(Sampler):
    def __init__(self):
        super().__init__()  # Call the parent constructor
    
    def sample(self, logits, prev_ids):
        """Sample from the logits specific to SynthIdSampler."""
        pass