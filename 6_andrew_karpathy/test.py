import torch
import matplotlib.pyplot as plt
words = open('data/names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
N = torch.zeros((27, 27), dtype=torch.int32)
check_list = list()
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        check_list.append((ix1, ix2))
    N[ix1, ix2] += 1
print(check_list)

print(N)


