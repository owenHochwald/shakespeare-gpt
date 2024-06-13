import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameter settings
batch_size = 32 # number of independent sequences to process in parallel
block_size = 8 # context length
max_iters = 3000 # number of times to train on the whole dataset
eval_interval = 300
learning_rate = 1e-2 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # using GPU if available
eval_iters = 200
# 32 dim embedding
n_embd = 32

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


# super simple bigram model, version 1
# here we take the code from the notebook and compile it into a class
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # encoding identities and position of tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # to go from tok_emb to logits, we need a linear layer
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        # ints from zero to T-1, which get embedded and summed up
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)\
        
        # x holds token identity and position which they occur
        x=tok_emb+pos_emb #(B,T,C)
        
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
            # getting all the predictions stored inside the lgoits
            logits, loss = self(idx)
            # focus only on the last element in the time step (predicting for the next token)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax multi class classification to get probabilities for most likely next token
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution and get one sample, changes dimensions
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # take ints from the sample and append to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
# when gpu is used, we need to move the model to the GPU so that calculations happen there
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# creating training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))