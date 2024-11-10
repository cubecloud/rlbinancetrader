
def calculate_indices(n_envs, n_processes):
    # Calculate the number of environments per process
    envs_per_process = n_envs // n_processes
    # Calculate the remaining environments
    remaining_envs = n_envs % n_processes
    # Initialize the list of indices
    indices = []
    # Initialize the start index
    start_idx = 0
    # Loop over the number of processes
    for i in range(n_processes):
        # Calculate the number of environments for this process
        num_envs = envs_per_process + (1 if i < remaining_envs else 0)
        # Calculate the end index
        end_idx = start_idx + num_envs
        # Append the indices for this process to the list
        env_lst = list(range(start_idx, end_idx))
        if env_lst:
            indices.append(list(range(start_idx, end_idx)))
        else:
            break
        # Update the start index
        start_idx = end_idx
    return indices

if __name__ == "__main__":
    print(calculate_indices(10, 15))
    print(calculate_indices(26, 15))
    print(calculate_indices(26, 1))