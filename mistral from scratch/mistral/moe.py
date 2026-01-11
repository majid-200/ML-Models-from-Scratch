import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

@dataclasses.dataclass
class MoeArgs(Serializable):
    """
    Configuration for Mixture of Experts (MoE) layer.
    
    Example: If num_experts=8 and num_experts_per_tok=2, each token will be
    processed by exactly 2 out of 8 available experts.
    """
    num_experts: int              # Total number of expert networks available
    num_experts_per_tok: int      # How many experts process each token (top-k)

class MoeLayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer - A Sparse Routing Mechanism
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    HOW MoE WORKS                            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Input Token -> Gate -> Select Top-K Experts -> Weighted Sum│
    │                                                             │
    │  Instead of using ALL experts (expensive), we:              │
    │  1. Use a "gate" network to score each expert               │
    │  2. Pick only the top-K experts with highest scores         │
    │  3. Combine their outputs with learned weights              │
    │                                                             │
    │  This makes the model SPARSE and much more efficient        │
    └─────────────────────────────────────────────────────────────┘
    
    Benefits:
    - Scales model capacity without proportional compute increase
    - Each expert can specialize in different patterns/domains
    - Only activates subset of parameters per token (sparse activation)
    """
    
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        
        # Store all expert networks (e.g., 8 different FFN networks)
        self.experts = nn.ModuleList(experts)
        
        # Gate network: decides which experts to use for each token
        # Typically a simple linear layer: input_dim → num_experts
        self.gate = gate
        
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass through the MoE layer.
        
        Args:
            inputs: Tensor of shape [num_tokens, hidden_dim]
                   (or [batch, seq_len, hidden_dim] flattened to 2D)
        
        Returns:
            Tensor of same shape as inputs, processed by selected experts
        
        
        ═══════════════════════════════════════════════════════════════
                            STEP-BY-STEP VISUALIZATION
        ═══════════════════════════════════════════════════════════════
        
        Assume: 4 tokens, 8 experts, top-2 experts per token
        
        STEP 1: GATE SCORING
        ────────────────────────────────────────────────────────────
        Each token gets a score for every expert:
        
        Token 0: [0.8, 0.2, 0.9, 0.1, 0.3, 0.7, 0.4, 0.6] → Pick top-2
        Token 1: [0.1, 0.9, 0.3, 0.8, 0.2, 0.4, 0.5, 0.6]
        Token 2: [0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 0.4, 0.5]
        Token 3: [0.4, 0.5, 0.2, 0.7, 0.6, 0.8, 0.3, 0.1]
        
        
        STEP 2: TOP-K SELECTION (K=2)
        ────────────────────────────────────────────────────────────
        Select the 2 highest-scoring experts for each token:
        
        Token 0 → Experts [2, 0]  (scores: 0.9, 0.8)
        Token 1 → Experts [1, 3]  (scores: 0.9, 0.8)
        Token 2 → Experts [4, 2]  (scores: 0.9, 0.8)
        Token 3 → Experts [5, 3]  (scores: 0.8, 0.7)
        
        
        STEP 3: SOFTMAX NORMALIZATION
        ────────────────────────────────────────────────────────────
        Normalize the selected scores so they sum to 1:
        
        Token 0: [0.9, 0.8] → softmax → [0.52, 0.48]
        Token 1: [0.9, 0.8] → softmax → [0.52, 0.48]
        Token 2: [0.9, 0.8] → softmax → [0.52, 0.48]
        Token 3: [0.8, 0.7] → softmax → [0.53, 0.47]
        
        
        STEP 4: EXPERT PROCESSING
        ────────────────────────────────────────────────────────────
        
        Expert 0 processes: Token 0          (weight: 0.48)
        Expert 1 processes: Token 1          (weight: 0.52)
        Expert 2 processes: Token 0, Token 2 (weights: 0.52, 0.48)
        Expert 3 processes: Token 1, Token 3 (weights: 0.48, 0.47)
        Expert 4 processes: Token 2          (weight: 0.52)
        Expert 5 processes: Token 3          (weight: 0.53)
        Expert 6 processes: (nothing)
        Expert 7 processes: (nothing)
        
        Notice: Not all experts are used! This is the "sparse" part.
        
        
        STEP 5: WEIGHTED COMBINATION
        ────────────────────────────────────────────────────────────
        Final output for each token is weighted sum:
        
        Token 0 = 0.52 * Expert_2(Token_0) + 0.48 * Expert_0(Token_0)
        Token 1 = 0.52 * Expert_1(Token_1) + 0.48 * Expert_3(Token_1)
        Token 2 = 0.52 * Expert_4(Token_2) + 0.48 * Expert_2(Token_2)
        Token 3 = 0.53 * Expert_5(Token_3) + 0.47 * Expert_3(Token_3)
        
        ═══════════════════════════════════════════════════════════════
        """
        
    
        # STEP 1: Gate Scoring
        # 
        # Generate scores for all experts for each token
        # Shape: [num_tokens, num_experts]
        # Each row contains scores indicating how suitable each expert is for that token
        gate_logits = self.gate(inputs)
        
        
        # STEP 2: Top-K Expert Selection                  
        # 
        # For each token, select the top `num_experts_per_tok` experts
        # weights: [num_tokens, num_experts_per_tok] - the top-k scores
        # selected_experts: [num_tokens, num_experts_per_tok] - indices of top-k experts
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        
    
        # STEP 3: Softmax Normalization                   
        # 
        # Apply softmax to the selected top-k scores (NOT all experts)
        # This ensures weights sum to 1.0 for each token
        # 
        # Why softmax AFTER top-k?
        # - Makes results consistent when changing num_experts or num_experts_per_tok
        # - Ensures fair comparison: weights always sum to 1 regardless of K
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        
        # Initialize output tensor (same shape as input)
        # We'll accumulate weighted expert outputs here
        results = torch.zeros_like(inputs)
        
    
        # STEP 4 & 5: Expert Processing + Combination     
        # 
        # Iterate through each expert and process its assigned tokens
        for current_expert_index, current_expert in enumerate(self.experts):
            
            # Find which tokens selected this expert
            # torch.where returns indices where condition is True
            # 
            # Example: If selected_experts = [[0, 2],    ← Token 0 uses experts 0 and 2
            #                                  [1, 3],    ← Token 1 uses experts 1 and 3
            #                                  [2, 0]]    ← Token 2 uses experts 2 and 0
            # 
            # When current_expert_index = 0:
            #   token_index = [0, 2]           ← Tokens 0 and 2 selected expert 0
            #   token_expert_index = [0, 1]    ← Expert 0 is at position 0 for token 0,
            #                                     and position 1 for token 2
            token_index, token_expert_index = torch.where(selected_experts == current_expert_index)
            
            # If no tokens selected this expert, skip it (sparse!)
            if len(token_index) == 0:
                continue
            
            # Process the selected tokens through this expert
            # Then weight the output and add to results
            # 
            # Breakdown:
            # 1. inputs[token_index] - Get input for tokens using this expert
            # 2. current_expert(...) - Process through expert network
            # 3. weights[token_index, token_expert_index, None] - Get corresponding weights
            #    (None adds dimension for broadcasting)
            # 4. Multiply weighted output and accumulate in results
            results[token_index] += weights[token_index, token_expert_index, None] * current_expert(
                inputs[token_index]
            )
        
        # Return the final weighted combination of all expert outputs
        return results


# ═══════════════════════════════════════════════════════════════════════
#                           KEY CONCEPTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════
#
# 1. CONDITIONAL COMPUTATION: Not all experts process all tokens
#    - More efficient than dense layers
#    - Allows model to scale without linear compute increase
#
# 2. LEARNED ROUTING: The gate network learns which experts to use
#    - Trained end-to-end with the rest of the model
#    - Experts can specialize (e.g., one for code, one for math)
#
# 3. SPARSE ACTIVATION: Only K out of N experts are active per token
#    - Example: 2 out of 8 experts = 25% activation rate
#    - Mistral models often use 8 experts, activate 2 per token
#
# 4. LOAD BALANCING: In practice, auxiliary losses ensure experts are
#    used roughly equally (not shown in this code)