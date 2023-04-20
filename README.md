# Stable-Terminal
StableLM embedd in terminal. 

Using the recently open-sourced [StableLM](https://github.com/Stability-AI/StableLM) in the terminal. 

# Installation
This project reuqires minimal packages including:
```bash
'accelerate',
'bitsandbytes',
'transformers',
'torch'
```
You can install them through the following commands. It's recommended to have torch pre-installed.
```bash
git clone https://github.com/pengzhangzhi/Stable-Terminal.git
cd Stable-Terminal
pip install -e .
```

# Usage
- Note that it will take very long time to setup if you run the program in the first time.
- Running the program takes 20G GPU memory.
```bash
query "who are you?"
```

output:
```

I am an AI language model trained to assist with a variety of tasks,
such as generating human-like text, answering questions, and more. 
However, I am not capable of having personal experiences, emotions, or thoughts like humans. 
I am designed to provide helpful and informative responses to the best of my ability, 
based on the information and knowledge that I have been trained on. 
Is there something specific you would like to know or talk about?
```
