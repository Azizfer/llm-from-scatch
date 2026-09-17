import torch
import torch.nn as nn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


checkpoint_path_to_test = '/kaggle/working/finetuned_checkpoint.pt'

raw_model = model.module if isinstance(model, nn.DataParallel) else model
ckpt = torch.load(checkpoint_path_to_test, map_location=device)
raw_model.load_state_dict(ckpt['model_state_dict'])
raw_model.to(device)
raw_model.eval()
print(f"Loaded checkpoint from iter {ckpt['iter']}")

prompt = "Question: Question: Tell me whether these cities are in Spain or France: Pamplona, Valencia, Nice, Marseille, Paris, Sevilla ? \n Answer: "
context = torch.tensor(encode(prompt) , dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    output = raw_model.generate(context, max_new_tokens=100, eos_token_id=eos_id)
print(decode(output[0].tolist()))