import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameter settings
batch_size = 64 # number of independent sequences to process in parallel
block_size = 256 # context length
max_iters = 5000 # number of times to train on the whole dataset
eval_interval = 500
# decreased learning rate, self-attention doesn't work with high LR
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # using GPU if available
eval_iters = 200
# 32 dim embedding
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


torch.manual_seed(1337)
# vocabulary size

# importing dataset
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text=f.read()
# developing vocab from the dataset
vocab = sorted(set(text))
vocab_size = len(vocab)
# creating the vocab lookup table from string to int
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
# encoding and decoding functions for this lookup table
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 
# training and validation dataset splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# data loading with batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    # memory effecient since tell pytorch we don't intend to do backprop
    out = {}
    # setting to evaluation mode
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # setting to training mode
    model.train()
    return out

class Head(nn.Module):
    """ creating one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        # creating layers for token identification
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # creating lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    # implementation of the blocks from the jupiter notebook    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # calculating attention scores ("likelihood of each token")
        # normalizing to get scaled attention scores
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        # decoder block: masking to make sure future tokens don't attend to past tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # softmax acts as a normalizer, sums elements in row and exponentiates them
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)
        # preforming weighted aggregation
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention block in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        # running heads in parallel in a list
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # adding projects
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        # concatenating self attention outputs over channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # applying project -> linear outcome of the Linear layer
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """feed forward function with linear layer followed by non-linear aspects"""
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # growing residual block layer by multiple of 4
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            # project layer for residual pathway
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    """Creating transformer block - token communication followed with computation for prediction"""
    
    def __init__(self,n_embd, n_head):
        # n_emd: embedding dimension
        # n_head: the number of desired heads - group size
        super().__init__()
        head_size = n_embd // n_head
        # communication
        self.sa = MultiHeadAttention(n_head, head_size)
        # computation done independently with feeding forward
        self.ffwd = FeedForward(n_embd)
        # normalizing layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

# bigram model with multiple heads of self attention
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # encoding identities and position of tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # decode layer
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # # better model init
        # self.apply(self._init_weights)
        
        # def _init_weights(self, module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #         if module.bias is not None:
        #             torch.nn.init.zeros_(module.bias)
        #     elif isinstance(module, nn.Embedding):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # ints from zero to T-1, which get embedded and summed up
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)\
        
        # x holds token identity and position which they occur
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        
        # decoder block to get logits
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # changing shape in order to feed into our loss function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            # b/c we're using positional embedding, we can never have more than the length of the block size
            idx_cond = idx[:,-block_size:]
            # getting all the predictions stored inside the lgoits
            logits, loss = self(idx_cond)
            # focus only on the last element in the time step (predicting for the next token)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax multi class classification to get probabilities for most likely next token
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution and get one sample, changes dimensions
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # take ints from the sample and append to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel(vocab_size)
# when gpu is used, we need to move the model to the GPU so that calculations happen there
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# creating training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass and evaluate the loss
    logits, loss = model(xb, yb)
    
    # backward pass
    # set gradients to zero for current pass
    optimizer.zero_grad(set_to_none=True)
    # backpropagate
    loss.backward()
    # update the parameters according to the optimizer function
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))