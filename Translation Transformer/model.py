import torch
import torch.nn as nn
import math

"""
TRANSFORMER ARCHITECTURE FROM "ATTENTION IS ALL YOU NEED"
========================================================

Overall Architecture:
    Input Sequence → [ENCODER] → Context Vector → [DECODER] → Output Sequence
    
    ENCODER: Processes the source sequence (e.g., English sentence)
    DECODER: Generates the target sequence (e.g., French translation)
"""

class LayerNormalization(nn.Module):
    """
    Layer Normalization: Normalizes inputs across features
    
    Purpose: Stabilizes training by normalizing the inputs to each layer
    
    How it works:
        1. Calculate mean and std across the feature dimension
        2. Normalize: (x - mean) / std
        3. Scale and shift with learnable parameters (alpha, bias)
    
    Shape transformation: (batch, seq_len, features) → (batch, seq_len, features)
    """

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable shift parameter

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        # Example: (32, 10, 512) = 32 sentences, 10 words each, 512 features per word
        
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch, seq_len, 1)
        
        # Normalize, scale, and shift
        # eps prevents division by zero
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Architecture:
        Input → Linear → ReLU → Dropout → Linear → Output
        
    Visual:
        (d_model) → [Linear + ReLU] → (d_ff) → [Linear] → (d_model)
          512    →                  →  2048  →           →    512
    
    Purpose: Adds non-linearity and processes each position independently
    Note: Same network applied to each position separately and identically
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # Expansion layer
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # Projection layer

    def forward(self, x):
        # Shape: (batch, seq_len, d_model) → (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        # Example: (32, 10, 512) → (32, 10, 2048) → (32, 10, 512)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class InputEmbeddings(nn.Module):
    """
    Token Embeddings: Converts token IDs to dense vectors
    
    Purpose: Transform discrete tokens into continuous vector representations
    
    Example:
        Token ID: 42 (word "hello")
        ↓
        Embedding lookup
        ↓
        Vector: [0.2, -0.5, 0.8, ..., 0.1]  (d_model dimensions)
    
    Note: Embeddings are scaled by sqrt(d_model) as per the paper
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len) - contains token IDs
        # Output: (batch, seq_len, d_model)
        # Example: (32, 10) → (32, 10, 512)
        
        # Multiply by sqrt(d_model) to scale embeddings (from the paper)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding: Adds position information to embeddings
    
    Why needed? Transformers have no inherent sense of token order (unlike RNNs)
    
    Formula (from paper):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Visual representation for a sequence:
        Position 0: [sin(θ₀), cos(θ₀), sin(θ₁), cos(θ₁), ...]
        Position 1: [sin(θ₀'), cos(θ₀'), sin(θ₁'), cos(θ₁'), ...]
        Position 2: [sin(θ₀''), cos(θ₀''), sin(θ₁''), cos(θ₁''), ...]
        
    These are ADDED to the embeddings, not concatenated
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        
        # Create division term for scaling
        # We want: 10000^(-2i/d_model) for i = 0, 1, 2, ..., d_model/2
        # 
        # Mathematical identity: a^b = exp(b * log(a))
        # So: 10000^(-2i/d_model) = exp(-2i/d_model * log(10000))
        #
        # Why use exp? More numerically stable for small numbers
        # 
        # Step by step:
        # 1. torch.arange(0, d_model, 2) → [0, 2, 4, 6, ..., d_model-2]  (even indices)
        # 2. Multiply by -log(10000)/d_model → [-0, -2*log(10000)/d_model, -4*log(10000)/d_model, ...]
        # 3. Apply exp → [1, 10000^(-2/d_model), 10000^(-4/d_model), ...]
        #
        # Result: [1, 0.912, 0.832, 0.759, ...]  (decreasing values)
        # This creates different "wavelengths" for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (seq_len, d_model) → (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # Add positional encoding to embeddings (broadcasting across batch)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization
    
    Architecture:
        x → LayerNorm → Sublayer → Dropout → (+) → Output
        ↓___________________________________|
        
    Formula: x + Dropout(Sublayer(LayerNorm(x)))
    
    Purpose:
        - Helps with gradient flow (prevents vanishing gradients)
        - Allows network to learn identity function easily
        - Enables training of very deep networks
    
    Note: This uses Pre-LN (Layer Norm before sublayer), which is more stable
    """
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # sublayer is a function (lambda) that will be applied to normalized x
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention: The core mechanism of the Transformer
    
    INTUITION:
        Attention allows each word to "look at" other words in the sequence
        and decide which ones are most relevant.
        
    EXAMPLE:
        Sentence: "The animal didn't cross the street because it was too tired"
        When processing "it", attention helps determine "it" refers to "animal"
    
    ARCHITECTURE:
        
        Input (d_model)
           ↓
        Split into h heads
           ↓
        Each head: [Q, K, V] = [Query, Key, Value]
           ↓
        Attention(Q,K,V) = softmax(QK^T / √d_k) × V
           ↓
        Concatenate heads
           ↓
        Linear projection
           ↓
        Output (d_model)
    
    WHY MULTI-HEAD?
        Different heads can learn different types of relationships:
        - Head 1: syntactic relationships (subject-verb)
        - Head 2: semantic relationships (synonyms)
        - Head 3: positional relationships (nearby words)
        etc.
    
    SHAPES EXAMPLE (d_model=512, h=8):
        Input:  (batch, seq_len, 512)
        Split:  (batch, seq_len, 8, 64)  [8 heads, 64 dims each]
        Rearrange: (batch, 8, seq_len, 64)
        Attention: (batch, 8, seq_len, 64)
        Concat: (batch, seq_len, 512)
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension (e.g., 512)
        self.h = h              # Number of attention heads (e.g., 8)
        
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h  # Dimension per head (e.g., 512/8 = 64)
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Scaled Dot-Product Attention
        
        Formula: Attention(Q,K,V) = softmax(QK^T / √d_k) × V
        
        STEP BY STEP:
        1. Compute similarity: Q × K^T (how much each word relates to others)
        2. Scale by √d_k (prevents softmax saturation (meaning it pushes probabilities very close to 0 or 1 (one-hot distributions)) for large d_k)
        3. Apply mask (hide future tokens in decoder, ignore padding)
        4. Softmax (convert to probabilities)
        5. Apply to values: multiply by V
        
        ATTENTION SCORES VISUALIZATION:
                    Key positions →
        Query pos ↓  [0.1, 0.7, 0.1, 0.1]  (word 1 attends mostly to word 2)
                     [0.3, 0.3, 0.3, 0.1]  (word 2 attends evenly)
                     [0.05, 0.05, 0.8, 0.1] (word 3 attends mostly to itself)
        """
        d_k = query.shape[-1]
        
        # Step 1 & 2: Compute attention scores and scale
        # Shape: (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) 
        #     → (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Set masked positions to very negative number (→ 0 after softmax)
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get probabilities
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Step 5: Apply attention to values
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) 
        #     → (batch, h, seq_len, d_k)
        output = attention_scores @ value
        
        return output, attention_scores

    def forward(self, q, k, v, mask):
        """
        q, k, v can be:
        - Same (self-attention): all come from the same sequence
        - Different (cross-attention): q from decoder, k,v from encoder
        """
        # Project inputs to Q, K, V
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)

        # Split into multiple heads and rearrange
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Concatenate heads back together
        # (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k) → (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear projection
        return self.w_o(x)
    
class EncoderBlock(nn.Module):
    """
    Single Encoder Layer
    
    ARCHITECTURE:
        Input
          ↓
        Multi-Head Self-Attention (with residual connection)
          ↓
        Feed-Forward Network (with residual connection)
          ↓
        Output
    
    Full detail:
        x → [LayerNorm → Self-Attention → Dropout] + x
          → [LayerNorm → Feed-Forward → Dropout] + x
    
    Purpose: Process input sequence and capture relationships between tokens
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections: one for attention, one for feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(2)
        ])

    def forward(self, x, src_mask):
        # Self-attention: query, key, value all come from the same input x
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        # Feed-forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Complete Encoder: Stack of N encoder layers
    
    ARCHITECTURE:
        Input Embeddings + Positional Encoding
          ↓
        Encoder Layer 1
          ↓
        Encoder Layer 2
          ↓
        ...
          ↓
        Encoder Layer N
          ↓
        Layer Normalization
          ↓
        Output (context representation)
    
    Purpose: Convert input sequence into rich context representation
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # Pass through all encoder layers sequentially
        for layer in self.layers:
            x = layer(x, mask)
        # Final layer normalization
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """
    Single Decoder Layer
    
    ARCHITECTURE:
        Input (target sequence)
          ↓
        Masked Multi-Head Self-Attention (with residual)
          ↓
        Multi-Head Cross-Attention to Encoder (with residual)
          ↓
        Feed-Forward Network (with residual)
          ↓
        Output
    
    KEY DIFFERENCE FROM ENCODER:
        1. Self-attention is MASKED (can't see future tokens)
        2. Cross-attention uses encoder output as K,V
    
    Full detail:
        x → [LayerNorm → Masked Self-Attention → Dropout] + x
          → [LayerNorm → Cross-Attention(Q=x, K=encoder, V=encoder) → Dropout] + x
          → [LayerNorm → Feed-Forward → Dropout] + x
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Three residual connections: self-attention, cross-attention, feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. Masked self-attention on target sequence
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        
        # 2. Cross-attention: query from decoder, key & value from encoder
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        
        # 3. Feed-forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Complete Decoder: Stack of N decoder layers
    
    ARCHITECTURE:
        Target Embeddings + Positional Encoding
          ↓
        Decoder Layer 1
          ↓
        Decoder Layer 2
          ↓
        ...
          ↓
        Decoder Layer N
          ↓
        Layer Normalization
          ↓
        Output
    
    Purpose: Generate output sequence using encoder context
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pass through all decoder layers sequentially
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Final layer normalization
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Final Projection Layer: Convert decoder output to vocabulary probabilities
    
    ARCHITECTURE:
        Decoder output (d_model) → Linear → Vocabulary logits (vocab_size)
    
    Shape: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
    
    Example:
        Input: (32, 10, 512)  [32 sentences, 10 tokens, 512 features]
        Output: (32, 10, 30000)  [32 sentences, 10 tokens, 30000 vocab scores]
    
    Note: Usually followed by softmax to get probabilities
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):
    """
    Complete Transformer Model
    
    FULL ARCHITECTURE:
    
        SOURCE                          TARGET
          ↓                               ↓
      Input Embed                    Input Embed
          ↓                               ↓
      Pos Encoding                   Pos Encoding
          ↓                               ↓
      ┌─────────┐                    ┌─────────┐
      │ ENCODER │ ────────────────→  │ DECODER │
      │         │   (context)        │         │
      │  N×     │                    │  N×     │
      │  - Attn │                    │  - Attn │
      │  - FFN  │                    │  - Cross│
      └─────────┘                    │  - FFN  │
                                     └─────────┘
                                          ↓
                                    Projection
                                          ↓
                                    Vocab Probs
    
    EXAMPLE USE CASE (Translation):
        Source: "Hello world" (English)
        Target: "Bonjour monde" (French)
        
        1. Encode "Hello world" → context vector
        2. Decode context → "Bonjour monde"
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """Encode source sequence into context representation"""
        # src: (batch, seq_len) - token IDs
        src = self.src_embed(src)     # (batch, seq_len, d_model)
        src = self.src_pos(src)        # Add positional encoding
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """Decode target sequence using encoder context"""
        # tgt: (batch, seq_len) - token IDs
        tgt = self.tgt_embed(tgt)      # (batch, seq_len, d_model)
        tgt = self.tgt_pos(tgt)        # Add positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """Project to vocabulary space"""
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                     src_seq_len: int, tgt_seq_len: int, 
                     d_model: int=512, N: int=6, h: int=8, 
                     dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Build a complete Transformer model
    
    PARAMETERS:
        src_vocab_size: Size of source vocabulary (e.g., 30000 English words)
        tgt_vocab_size: Size of target vocabulary (e.g., 30000 French words)
        src_seq_len: Maximum source sequence length (e.g., 100)
        tgt_seq_len: Maximum target sequence length (e.g., 100)
        d_model: Embedding dimension (paper uses 512)
        N: Number of encoder/decoder layers (paper uses 6)
        h: Number of attention heads (paper uses 8)
        dropout: Dropout rate (paper uses 0.1)
        d_ff: Feed-forward hidden dimension (paper uses 2048)
    
    TYPICAL VALUES (from paper):
        d_model = 512
        N = 6
        h = 8
        d_ff = 2048
        dropout = 0.1
    """
    
    # Create embedding layers (convert token IDs to vectors)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers (add position information)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create N encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, 
                                    feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create N decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, 
                                    decoder_cross_attention_block, 
                                    feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Assemble encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create projection layer (convert to vocabulary probabilities)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Assemble complete transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, 
                            src_pos, tgt_pos, projection_layer)
    
    # Initialize parameters using Xavier uniform initialization
    # This helps with training stability
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

"""
USAGE EXAMPLE:
==============

# Build model
model = build_transformer(
    src_vocab_size=30000,  # English vocabulary
    tgt_vocab_size=30000,  # French vocabulary
    src_seq_len=100,       # Max source length
    tgt_seq_len=100,       # Max target length
)

# Training
src = torch.tensor([[1, 2, 3, 4, 5]])  # "Hello world" (token IDs)
tgt = torch.tensor([[10, 11, 12, 13]]) # "Bonjour monde" (token IDs)

encoder_output = model.encode(src, src_mask=None)
decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt, tgt_mask=None)
logits = model.project(decoder_output)  # (batch, seq_len, vocab_size)

# Apply softmax to get probabilities
probs = torch.softmax(logits, dim=-1)

# Get predicted tokens
predicted_tokens = torch.argmax(probs, dim=-1)
"""