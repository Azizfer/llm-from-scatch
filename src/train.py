import math, os
import torch

max_iterations=17000
max_lr = 2e-4
min_lr = 2e-5
warmup_iters = 150
eval_iters=100
lr_decay_iters = max_iterations
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


input_checkpoint_path = '/kaggle/input/models/azizferchichi/gbt/pytorch/default/1/gpt_checkpoint (2).pt'  # read-only, load FROM here
checkpoint_path = '/kaggle/working/gpt_checkpoint.pt' 
@torch.no_grad() # In here we are desactivating gradients in order to reduce computation because we only needed to get losses values only

def estimate_loss():
  out = {}
  model.eval()   # In evaluation mode everything would be active so even the random neurons thats why "Dropout" is desactivated here
  for split in ['train', 'val']:
     losses = torch.zeros(eval_iters)
     for k in range(eval_iters):
       X, Y = get_batch(split)
       logits, loss = model(X, Y)
       loss = loss.mean() 
       losses[k] = loss.mean()
       out[split] = losses.mean()

  model.train() # In Training mode 'Dropout' is active which helps the model to get trained by droping the unineeded neurons to prevent overfitting for example
  return out


def get_lr(it, resuming=False):
    effective_warmup = 0 if resuming else warmup_iters
    if it < effective_warmup: #Warmup
        return max_lr * (it + 1) / effective_warmup
    if it > lr_decay_iters: 
        return min_lr
    decay_ratio = (it - effective_warmup) / (lr_decay_iters - effective_warmup)  #Decaying concept
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

model.to(device)

raw_model_for_loading = model.module if isinstance(model, nn.DataParallel) else model
ckpt = torch.load(checkpoint_path, map_location=device)
raw_model_for_loading.load_state_dict(ckpt['model_state_dict'])
start_iter = ckpt['iter'] + 1
print(f"Resumed from iter {start_iter}")

if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
    model = nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

for iter in range(start_iter, max_iterations):  
    lr = get_lr(iter, resuming=True)   # If the model is already trained ( resume = True ) we skip the warmup 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
        torch.save({
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),  
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter,
        }, checkpoint_path) 
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    loss = loss.mean()         
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
print(loss.mean())
# Final save
torch.save({
    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'iter': iter,
}, checkpoint_path)  
print("model saved")