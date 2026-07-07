# Tutorial: Training on the DIIR Compute Node

This tutorial provides a step-by-step workflow for running LLM experiments on the highly secure, offline DIIR compute nodes.

The DIIR compute nodes are completely disconnected from the external internet for security reasons. You cannot run `pip install`, `git clone`, or download models directly from HF during your job. All environments and dependencies must be packed beforehand.

## Step 1: The Server Architecture

The DIIR infrastructure consists of three main components. Understanding their roles is important for navigating the offline environment. All the 3 servers require CUHK VPN connection.

1. **CHPC Server (The Command Center)**

   * **Purpose:** This is our login node (we use aiss compute node on this node). We use this server to submit, manage, and cancel our SLURM jobs (using commands like `sbatch` and `squeue`).

2. **137.189.4.221 Server (The Transition Station)**

   * **Purpose:** Since the compute nodes lack internet access, this server acts as a secure gateway. You will use it to upload your code, Conda environments, and Docker/Podman containers to the DIIR isolated environment. The `to_Secure_Store` folder is used to transfer the files from **137.189.4.221 to Diir computation node 137.189.4.220.**

3. **NoMachine Server - 137.189.4.220 (The Visual GUI)**

   * **Purpose:** Accessible via NoMachine or via a web browser at `https://137.189.4.220:4443/`. This provides a virtual Linux desktop. You can use it to visually monitor your code's progress, browse generated images/graphs, and check output logs in real-time.

## Step 2: Packing Your Code and Conda Environment on CHPC node or local pc

Because the DIIR node is offline, you must pack your environment and code on the aiss folder in CHPC node or your local pc machine beforehand.

### Packing the Conda Environment

I recommend using `conda-pack` to bundle your entire Python environment into a single portable archive.

```bash
# 1. Install conda-pack in your base environment
conda install -c conda-forge conda-pack

# 2. Pack your target environment (e.g., 'segr1_env')
conda pack -n segr1_env -o segr1_env_packed.tar.gz
```

### Packing Your Code

Compress your working directory containing your training scripts.

```bash
tar -czvf sam_llm_grpo_code.tar.gz sam_llm_grpo/
```

## Step 3: Packing the Docker Container Locally

The computation node runs jobs inside Podman/Docker containers. You **must** build and export this container on a local pc machine first (we do not have the permission to creating docker container on the CHPC node). **Please ensure you have Docker Desktop installed on your local machine before proceeding (https://www.docker.com/products/docker-desktop/).** I recommend using the official Hugging Face GPU image as your base (**https://hub.docker.com/r/huggingface/transformers-pytorch-gpu**).

### 1. Create a Dockerfile

Create a file named `Dockerfile` (with **no** file extenstion like .txt or .py) on your local pc machine. U can use notepad to create a Dockerfile.txt with the following contents and then delete the file extention .txt:

```dockerfile
# Use the Hugging Face PyTorch GPU image as the base
FROM huggingface/transformers-pytorch-gpu:latest

# Install any additional system-level dependencies required by your code
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /working-dir
```

### 2. Build and Save the Image

Open **Docker Desktop.** cd to the local dictionary where u save the `Dockerfile`. Run the following commands locally via Powershell or Terminal to build the image and save it as a `.tar` file:

```bash
# Build the image
docker build -t grpo:v1 .

# Save the image to a tar archive (This file will be large! Around 26 Gb if u use HF GPU Transformer image)
docker save -o grpo_image.tar grpo:v1 # u need to write down this name and tag (grpo:v1) because this will be used in creating the slurm file.
```

## Step 4: Uploading to the 221 Transition Server

Now, transfer your packed environment, code, and container to the DIIR secure zone.

1. **Upload files to the 221 server** using `scp`:

   ```bash
   scp segr1_env_packed.tar.gz sam_llm_grpo_code.tar.gz grpo_image.tar username@137.189.4.221:~
   ```

2. **Move files to the Secure Store:**
   SSH into the `137.189.4.221` server, locate the `to_Secure_Store` folder, and move your files into it:

   ```bash
   mv segr1_env_packed.tar.gz sam_llm_grpo_code.tar.gz grpo_image.tar ~/to_Secure_Store/
   ```
<img width="2152" height="1336" alt="image" src="https://github.com/user-attachments/assets/39c2b694-c666-4ebe-9491-c3c4cc1be36f" />

Wait a few minutes, and the system will automatically transport these files into your private directory (usually `/diir/$USER/private/`) on the DIIR compute nodes.

## Step 5: Checking Upload Status on NoMachine UI

Before writing and submitting your SLURM script, you should verify that your files have been successfully synced to the DIIR computation node.

Login to the NoMachine **Server (137.189.4.220)**, and use the following commands (OR DIRECTLY CHECK IN NOMACHINE UI WITH THE FOLLOWING STEPS):

```bash
# 1. Mount your DIIR network drive
mountdiir

# 2. Check if your files exist in your private folder or Directly check via NoMachine UI
ls -lh /diir/$USER/private/

# 3. Unmount the drive (IMPORTANT: Always unmount after checking)
umountdiir
```
<img width="1338" height="896" alt="image" src="https://github.com/user-attachments/assets/51ed5018-d1f2-4e42-ba1d-08cb7bfd8893" />
<img width="1350" height="892" alt="image" src="https://github.com/user-attachments/assets/1809f8d4-8232-4534-ae42-728bcbbacc4e" />
<img width="1336" height="892" alt="image" src="https://github.com/user-attachments/assets/a7080636-8722-4c7b-aef2-a0cace35545d" />
<img width="1336" height="882" alt="image" src="https://github.com/user-attachments/assets/21cbcc98-30ee-4167-ae43-4cd657afd0be" />

<img width="1348" height="900" alt="image" src="https://github.com/user-attachments/assets/2ea6f263-66c0-4c3b-a232-809627e0e495" />
u can see your containers, datasets, codes, and conda env here:
<img width="1326" height="892" alt="image" src="https://github.com/user-attachments/assets/49249814-bc9a-4ebc-ac6f-eada79e6a012" />
<img width="1333" height="889" alt="image" src="https://github.com/user-attachments/assets/746134bb-3d62-4776-9456-973638f97ede" />




## Step 6: Writing the SLURM Script on CHPC node

Once the files are synced, you need a SLURM script to orchestrate the loading of the container, unpacking of the environment, and execution of the training.

Create a file named `your_exp.slurm` on the **CHPC Server**.

***Crucial Note on Paths:*** Pay special attention to the volume mount line in the script: `-v /diir/$USER/private/sam_llm_grpo:/working-dir/code`. This maps your physical folder on the server to `/working-dir/code` inside the Podman container. Therefore, you **must** change all absolute paths in your Python scripts and SLURM arguments to match this internal path.

* **Example:** If your dataset is physically located at `/diir/s1155191391/private/sam_llm_grpo/datasets/data.jsonl`, inside the SLURM script you must pass it as `--dataset_jsonl /working-dir/code/datasets/data.jsonl`. In the python script, u need to change the absolute path from `/diir/s1155191391/private/sam_llm_grpo/datasets/data.jsonl` to `/working-dir/code/datasets/data.jsonl`.

This is my SLURM script example to run grpo:

```bash
#!/bin/bash
#SBATCH --job-name=Regional_GRPO
#SBATCH --account=diir
#SBATCH --partition=diir
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/diir/s1155191391/log/%j.out
#SBATCH --error=/diir/s1155191391/log/%j.err

mountdiir

USER=s1155191391
HOME=/diir/$USER/podman/
XDG_RUNTIME_DIR=/diir/$USER/podman/.local/share/containers/storage
mkdir -p $XDG_RUNTIME_DIR

echo "Cleaning up corrupted Podman state from previous crashes..."
podman system migrate || true
rm -rf /diir/$USER/podman/.local/share/containers/storage/libpod/tmp/* || true

echo "Loading offline base image..."
podman load -i /diir/$USER/private/grpo_image.tar

echo "Starting Container for Distributed GRPO..."

podman run \
    --device [nvidia.com/gpu=all](https://nvidia.com/gpu=all) \
    --security-opt=label=disable \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /diir/$USER/private:/my_private \
    -v /diir/$USER/private/sam_llm_grpo:/working-dir/code \
    -v /diir/$USER/private/cache:/working-dir/cache \
    docker.io/library/grpo:v1 \
    bash -c "\
        export TRITON_CACHE_DIR='/tmp/triton_cache' && \
        export HF_HOME='/working-dir/cache' && \
        export TRANSFORMERS_CACHE='/working-dir/cache' && \
        export TORCH_HOME='/working-dir/cache' && \
        export VLLM_CACHE_DIR='/working-dir/cache' && \
        export HF_DATASETS_OFFLINE=1 && \
        export TRANSFORMERS_OFFLINE=1 && \
        export OMP_NUM_THREADS=4 && \
        export MKL_NUM_THREADS=4 && \
        export OPENBLAS_NUM_THREADS=4 && \
        export VECLIB_MAXIMUM_THREADS=4 && \
        export NUMEXPR_NUM_THREADS=4 && \
        export WANDB_API_KEY='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' && \
        export WANDB_MODE='offline' && \
        export CUDA_VISIBLE_DEVICES='0,1,2,3' && \
        export DEBUG_MODE='true' && \
        \
        echo 'Extracting Conda Environment...' && \
        mkdir -p /opt/segr1_env && \
        tar -xzf /my_private/segr1_env_packed.tar.gz -C /opt/segr1_env --no-same-owner && \
        echo 'Activating Environment...' && \
        source /opt/segr1_env/bin/activate && \
        conda-unpack && \
        \
        echo 'Hot-patching vLLM Qwen2.5-VL dtype bug...' && \
        sed -i 's/self.visual(pixel_values, grid_thw/self.visual(pixel_values.to(dtype=torch.bfloat16), grid_thw/g' /opt/segr1_env/lib/python3.10/site-packages/vllm/model_executor/models/qwen2_5_vl.py && \
        \
        echo 'Hot-patching Transformers Qwen2.5-VL Rotary Embedding bug...' && \
        sed -i 's/\.float(), cos, sin/, cos, sin/g' /opt/segr1_env/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py && \
        \
        echo 'Initializing Distributed Torchrun for Regional GRPO...' && \
        cd /working-dir/code && \
        mkdir -p /working-dir/code/Seg-R1/GRPO_logs_07_04_regional/exp_grpo && \
        \
        torchrun \
            --nproc_per_node='2' \
            --nnodes='1' \
            --node_rank='0' \
            --master_addr='127.0.0.1' \
            --master_port='12345' \
            /working-dir/code/Seg-R1/seg-r1/src/open_r1/grpo_report_as_reward_07_04_regional.py \
            --use_vllm true \
            --output_dir /working-dir/code/Seg-R1/GRPO_logs_07_04_regional/exp_grpo \
            --model_name_or_path /working-dir/code/Seg-R1/SFT_logs/Qwen2.5-VL-3B-Instruct-Seg-R1-Cold-Start_ver2_05_23_combined \
            --dataset_name COD \
            --dataset_jsonl /working-dir/code/datasets/Medical_Dataset_ver16_shared_grpo_sft/GRPO/grpo_dataset_with_alloc.jsonl \
            --max_prompt_length 4096 \
            --max_completion_length 1024 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-6 \
            --lr_scheduler_type 'constant' \
            --logging_steps 1 \
            --bf16 True \
            --gradient_checkpointing true \
            --attn_implementation flash_attention_2 \
            --num_train_epochs 2 \
            --run_name Seg-R1-Regional-GRPO \
            --save_steps 500 \
            --save_total_limit 10 \
            --save_only_model true \
            --report_to wandb \
            --temperature 1.0 \
            --num_generations 4 \
            --vllm_device 'cuda:2' \
            --sam_device 'cuda' \
            --vllm_gpu_memory_utilization 0.6 \
            --sam_checkpoint /working-dir/code/Seg-R1/third_party/sam2/experiment_logs_05_23_combine/checkpoints_lung/checkpoint.pt \
            --deepspeed /working-dir/code/Seg-R1/seg-r1/local_scripts/zero1_no_optimizer.json \
            --dataloader_num_workers 1 \
            --dataloader_pin_memory false \
            2>&1 | tee /working-dir/code/Seg-R1/GRPO_logs_07_04_regional/exp_grpo/training_log.txt \
    "

umountdiir
echo "GRPO Training Job Done!"
```

### U can use this simple structure:

```bash
#!/bin/bash
#SBATCH --job-name=My_General_Job
#SBATCH --account=diir
#SBATCH --partition=diir
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/diir/%u/log/%j.out
#SBATCH --error=/diir/%u/log/%j.err

mountdiir

USER=$(whoami)
HOME=/diir/$USER/podman/
export XDG_RUNTIME_DIR=/diir/$USER/podman/.local/share/containers/storage
mkdir -p $XDG_RUNTIME_DIR

echo "Loading offline base image..."
podman load -i /diir/$USER/private/your_image.tar

echo "Starting General Python Job..."

podman run \
    --device [nvidia.com/gpu=all](https://nvidia.com/gpu=all) \
    --security-opt=label=disable \
    --ipc=host \
    -v /diir/$USER/private:/my_private \
    -v /diir/$USER/private/my_code_folder:/working-dir/code \
    docker.io/library/grpo:v1 \
    bash -c "\
        echo 'Extracting Conda Environment...' && \
        mkdir -p /opt/my_env && \
        tar -xzf /my_private/my_env_packed.tar.gz -C /opt/my_env --no-same-owner && \
        source /opt/my_env/bin/activate && \
        conda-unpack && \
        \
        echo 'Running python script...' && \
        cd /working-dir/code && \
        python main.py \
    "

umountdiir
echo "Job Done!"
```

## Step 7: Submitting the Job on the CHPC node

Login to the **CHPC server** and navigate to the directory to your home.
```bash
cd ~
```

Submit the job using `sbatch`:

```bash
sbatch absolute_path_to_slurm/run_regional_grpo.slurm
```

To check the status of your job (whether it is Pending `PD` or Running `R`):

```bash
squeue -u $USER
```

## Step 8: Monitoring Job Progress

You can monitor your training progress directly through the NoMachine visual interface:

1. Open NoMachine or Open your web browser and go to `https://137.189.4.220:4443/`.
2. Log in with DUO and your password.
3. Open the file explorer (folder icon) within the NoMachine Linux desktop.
4. Navigate to your log directory (e.g., `/diir/<your_username>/private/log/`).
5. Simply double-click to open the generated `.err` and `.out` files. You can refresh them to view your code's real-time outputs and debugging information.
<img width="1342" height="889" alt="image" src="https://github.com/user-attachments/assets/a9e29edf-1c3a-409e-a0fe-52be49ac0b96" />
<img width="1344" height="892" alt="image" src="https://github.com/user-attachments/assets/ccd734c6-8bd2-47bc-994d-37962f6e79a9" />
<img width="1336" height="889" alt="image" src="https://github.com/user-attachments/assets/16d501d4-718c-461d-8af0-4703e771d3f0" />

