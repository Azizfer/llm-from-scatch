import torch.nn.functional as F # gives access to neural net related functions
import torch.nn as nn
import torch
import pickle
import math 

batch_size=96
block_size=128 #Sequence length
learning_rate=1.5e-4 # Represents how far we go into the direction determined by the gradient
max_iterations=13500
eval_iters=100
dropout =0.1
n_embd=512
n_layer=12
n_head=8

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    current_ds = train_ds if split == "train" else test_ds

    x_batch = []
    y_batch = []

    # Collect `batch_size` samples for the current batch
    for _ in range(batch_size):
        while True:
            doc_idx = torch.randint(0, len(current_ds), (1,)).item()
            document_text = current_ds[doc_idx]['text']

            document_tokens = encode(document_text)

          
            if len(document_tokens) >= block_size + 1:

                start_pos = torch.randint(
                    0, len(document_tokens) - block_size, (1,)
                ).item()

                context_ids = document_tokens[start_pos : start_pos + block_size]
                target_ids = document_tokens[start_pos + 1 : start_pos + 1 + block_size]

                x_batch.append(torch.tensor(context_ids, dtype=torch.long))
                y_batch.append(torch.tensor(target_ids, dtype=torch.long))
                break  

    # Stack the list of individual tensors into single batch tensors
    X = torch.stack(x_batch)
    Y = torch.stack(y_batch)

    return X.to(device), Y.to(device)

x_test, y_test = get_batch("train")
print(f"Sample X batch shape: {x_test.shape}")
print(f"Sample Y batch shape: {y_test.shape}")

class Head (nn.Module) :
  def __init__(self, head_size) :
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False) # what do I have to offer
    self.query = nn.Linear(n_embd, head_size, bias=False) # Query — "what am I looking for?
    self.value = nn.Linear(n_embd, head_size, bias=False) # what do I actually hand over, if you decide I'm relevant?
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # A matrix of 1s, and 0s above diag ensuring that we only look on the previous tokens only

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (Batch, Block-size, channels) --> output of size (Batch, Block-size, head-size)
    B,T,C = x.shape
    k = self.key(x) # (B,T,head-size)
                                        #==> Every token in every sequence in the batch now has its own Query vector and Key vector, of dimension head_size
    q= self.query(x) # (B,T,head-size)
    wei = q @ k.transpose(-2,-1)* k.shape[-1] **- 0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    # For every pair of positions (i, j), it's the dot product of token i's query with token j's key .
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Scaling because the dot product can result into getting large values
    wei = F.softmax(wei, dim =- 1) # Now wei[b, i, :] reads as: "token i's attention distribution over all previous tokens (including itself)." E.g. [0.7, 0.2, 0.1, 0, 0, ...]
    wei = self.dropout (wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,hs)
    out = wei @ v # If token i put 70% weight on token 0, then 70% of token 0's "content(value)" flows into token i's output.
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj=nn.Linear(head_size*num_heads,n_embd)
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    out=torch.cat([h(x) for h in self.heads],dim=-1)
    out=self.dropout(self.proj(out))
    return out
class FeedForward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.net=nn.Sequential(
        nn.Linear(n_embd,4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd,n_embd),
        nn.Dropout(dropout) # To prevent overfitting as it desactivate some neurons
    )
  def forward(self,x):
    return self.net(x)

class Block(nn.Module): # This is one Decoder Layer which does 2 things : Apply self-attention + normalisation
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size) # Self attention (Each word asks) : Given these words what word should i pay attention to
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    y = self.sa(x)
    x = self.ln1(x + y) # Add + Norm 1
    y =self.ffwd(x)
    x= self.ln2(x+y)   # Add + Norm 2
    return x
class GPTLanguageModel(nn.Module):  #nn.Module gives us acess to all the needed functionds and utilities
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table= nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table=nn.Embedding(block_size,n_embd)
    self.blocks=nn.Sequential(*[Block(n_embd,n_head=4) for _ in range(n_layer)])
    self.ln_f=nn.LayerNorm(n_embd)
    self.lm_head=nn.Linear(n_embd,vocab_size)
    self.apply(self.init_weights)

  def init_weights(self, module):
    if isinstance(module, nn.Linear): #This initialisation is for the neural net neurons
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias) #The bias is universally set to 0 because at x=0 bias is the one that swtiches the decision boundary
    elif isinstance(module, nn.Embedding): #Initialisation of the embd table weights
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self,index,targets=None):
    tok_emb=self.token_embedding_table(index)
    # Ensure position_embedding_table input is on the same device as the model
    pos_emb=self.position_embedding_table(torch.arange(index.shape[1], device=index.device))
    x=tok_emb+pos_emb #This is a combo of each token embedded + its position
    x=self.blocks(x)
    x=self.ln_f(x)
    logits=self.lm_head(x)

    if targets is None :
      loss = None
    else :
      B , T , C = logits.shape
      logits=logits.view(B*T,C)
      targets=targets.view(B*T)
      loss=F.cross_entropy(logits,targets)
    return logits , loss

  def generate(self, index, max_new_tokens, top_k=40, top_p=0.9 , eos_token_id=None):
    for _ in range(max_new_tokens):
        index_cond = index[:, -block_size:]
        logits, loss = self(index_cond)
        logits = logits[:, -1, :]

        # Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-P (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        index_next = torch.multinomial(probs, num_samples=1)
        index = torch.cat((index, index_next), dim=1)
    return index

model=GPTLanguageModel(vocab_size=enc.n_vocab)
