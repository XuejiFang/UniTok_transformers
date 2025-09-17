from UniTok import UniTok
import torch
model_path = "ckpt/UniTok_transformers" # "XuejiFang/UniTok_transformers"

unitok = UniTok.from_pretrained(model_path)
states = torch.load("ckpt/384-4/ckpt-ep0-iter27035.pth", map_location="cpu")['trainer']['unitok']
unitok.load_state_dict(states, strict=True)

unitok.save_pretrained("ckpt/384-4/")

