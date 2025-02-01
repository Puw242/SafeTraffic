# TrafficSafe

![TrafficSafe Icon](assets/TrafficSafe_icon.gif)

## Set up environment
We conduct experiments under CUDA 12.1 and Ubuntu 22.04 on Nvidia A100 GPU.

Create conda environment
```
conda create -n TrafficSafe python=3.9
source activate TrafficSafe
```
Install the related packages.
```
pip install -r requirements.txt
```

## Train TrafficSafe LLM
Run the following command to train an 8B Llama-3.1 model on the Illinois (IL) dataset to predict crash severity:
```
bash train_TrafficSafe.sh -s 8B -t Llama-3.1 -d IL -p severity -i localhost:0,1 -m 50001 -o "./"
```

It will take around 150 mins using 2 Nvidia A100 GPUs to complete the fine-tuning process.
