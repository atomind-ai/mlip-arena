from huggingface_hub import hf_hub_download

fpath = hf_hub_download(repo_id="cyrusyc/mace-universal", subfolder="pretrained", filename="2023-12-12-mace-128-L1_epoch-199.model")

print(fpath)