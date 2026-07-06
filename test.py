from tqdm import tqdm

pdm = tqdm(range(1, 10000), ncols=100)
for n in pdm : 
    print(n)