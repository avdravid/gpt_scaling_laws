# Scaling Laws of Neural Language Models

## Set-Up
How to run: ``torchrun --nproc_per_node=XXXX train.py config/scale_gpt.py --scale_N=True --n_layer=XXXX --n_embd=YYYY --out_dir="ZZZZZZZ"``

Original repo only goes up to a few million parameters. Let's scale it up. 
1) original GPT: n_layer=12 n_embd=768
2) GPT medium: n_layer=24, n_embd=1024
3) GPT large: n_layer=36, n_embd=1280
4) GPT XL: n_layer=48, n_embd=1600

   
  
