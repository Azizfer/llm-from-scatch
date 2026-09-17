import tiktoken
enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return enc.encode(text , allowed_special={"|<endoftext>|"})

def decode(tokens):
    return enc.decode(tokens)git 