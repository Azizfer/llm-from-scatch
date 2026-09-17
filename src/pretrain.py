from datasets import load_dataset
import torch.nn as nn
import torch

block_size=96
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

raw_model = model.module if isinstance(model, nn.DataParallel) else model
raw_model.to(device)


checkpoint_path = '/kaggle/working/finetuned_checkpoint.pt'  


print(" Loading previous fine-tuned checkpoint...")
ckpt = torch.load(checkpoint_path, map_location=device)


state_dict = ckpt['model_state_dict']
if list(state_dict.keys())[0].startswith('module.'):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v  
    state_dict = new_state_dict

raw_model.load_state_dict(state_dict)
print(f" Loaded checkpoint from step {ckpt.get('iter', 'unknown')}")


optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-5)

dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_example(example):
    instruction = example['instruction']
    context = example['context']
    response = example['response']
    if context and context.strip():
        prompt = f" Question: {instruction}\nContext: {context}\n Answer: "
    else:
        prompt = f" Question: {instruction}\n Answer: "
    return prompt + response + "<|endoftext|>"

eos_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

qa_tokenized = [
    enc.encode(format_example(ex), allowed_special={"<|endoftext|>"}) 
    for ex in dolly
]
qa_tokenized = [t for t in qa_tokenized if len(t) >= 2]

def get_qa_batch(batch_size):
    x_batch, y_batch = [], []
    for _ in range(batch_size):
        while True:
            tokens = qa_tokenized[torch.randint(0, len(qa_tokenized), (1,)).item()]
            if len(tokens) >= block_size + 1:
                start = torch.randint(0, len(tokens) - block_size, (1,)).item()
                context_ids = tokens[start : start + block_size]
                target_ids = tokens[start + 1 : start + 1 + block_size]
            else:
                pad_len = block_size + 1 - len(tokens)
                padded = tokens + [eos_id] * pad_len
                context_ids = padded[:block_size]
                target_ids = padded[1:block_size + 1]
            x_batch.append(torch.tensor(context_ids, dtype=torch.long))
            y_batch.append(torch.tensor(target_ids, dtype=torch.long))
            break
    return torch.stack(x_batch).to(device), torch.stack(y_batch).to(device)


finetune_steps = 3000
for step in range(finetune_steps):
    xb, yb = get_qa_batch(batch_size=32)
    logits, loss = raw_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step}: loss {loss.item():.4f}")


torch.save({
    'model_state_dict': raw_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  
    'iter': finetune_steps,
}, '/kaggle/working/finetuned_checkpoint.pt')
print(" Model and optimizer saved!")