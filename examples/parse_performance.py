import os

llm_ranks = [64]  # Change this according to your needs
ve_ranks = [1, 4, 8, 16, 24, 32, 64]  # Don't change this list
ve_print = [1, 4, 8, 16, 24, 32, 64]  # Don't change this list
dataset_sizes = [2**i for i in range(3, 18)]  # print perplexity from 2^3 to 2^17

base_dir = "/workspace/Cornstarch/examples"  # Path where log dirs are located

results = {}

for llm_rank in llm_ranks:
    for ve_rank in ve_ranks:
        for dataset_size in dataset_sizes:
            dir_name = f"LLaVA-OV-7B-rVE{ve_rank}_rLLM{llm_rank}_{dataset_size}_samelr"
            dir_path = os.path.join(base_dir, dir_name)
            log_file_path = os.path.join(dir_path, "log.log")

            if os.path.isfile(log_file_path):
                with open(log_file_path, "r") as f:
                    for line in f:
                        if "Epoch 1 perplexity:" in line:
                            value = line.strip().split()[-1]
                            results[dir_name] = value

                            break
                    else:
                        print(f"No 'Epoch1 perplexity' line found in {log_file_path}")
            else:
                print(f"File not found: {log_file_path}")
                break


print("*" * 50)
print("[Ep1] llm_rank:", llm_rank)
print("*" * 50)
for r_ve in ve_print:
    print(r_ve, end=": ")
    for key, value in results.items():
        if f"rVE{r_ve}_" in key:
            print(value, end=" ")
    print()

results = {}

for llm_rank in llm_ranks:
    for ve_rank in ve_ranks:
        for dataset_size in dataset_sizes:
            dir_name = f"LLaVA-OV-7B-rVE{ve_rank}_rLLM{llm_rank}_{dataset_size}_samelr"
            dir_path = os.path.join(base_dir, dir_name)
            log_file_path = os.path.join(dir_path, "log.log")

            if os.path.isfile(log_file_path):
                with open(log_file_path, "r") as f:
                    for line in f:
                        if "Epoch 2 perplexity:" in line:
                            value = line.strip().split()[-1]
                            results[dir_name] = value

                            break
                    else:
                        print(f"No 'Epoch2 perplexity' line found in {log_file_path}")
            else:
                print(f"File not found: {log_file_path}")
                break


print("*" * 50)
print("[Ep2] llm_rank:", llm_rank)
print("*" * 50)
for r_ve in ve_print:
    print(r_ve, end=": ")
    for key, value in results.items():
        if f"rVE{r_ve}_" in key:
            print(value, end=" ")
    print()


results = {}

for llm_rank in llm_ranks:
    for ve_rank in ve_ranks:
        for dataset_size in dataset_sizes:
            dir_name = f"LLaVA-OV-7B-rVE{ve_rank}_rLLM{llm_rank}_{dataset_size}_samelr"
            dir_path = os.path.join(base_dir, dir_name)
            log_file_path = os.path.join(dir_path, "log.log")

            if os.path.isfile(log_file_path):
                with open(log_file_path, "r") as f:
                    for line in f:
                        if "Epoch 3 perplexity:" in line:
                            value = line.strip().split()[-1]
                            results[dir_name] = value
                            break
                    else:
                        print(f"No 'Epoch3 perplexity' line found in {log_file_path}")
            else:

                print(f"File not found: {log_file_path}")
                break

print("*" * 50)
print("[Ep3] llm_rank:", llm_rank)
print("*" * 50)
for r_ve in ve_print:
    print(r_ve, end=": ")
    for key, value in results.items():
        if f"rVE{r_ve}_" in key:
            print(value, end=" ")
    print()
