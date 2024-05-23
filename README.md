## LLM_fine_tuning_llama3_8b

### Introduction

The repo is a demonstration of fine tuning an open-source LLM (Llama-3-8B) utilizing different approaches and techniques. Fine-Tuning was done with ORPO technique that combines SFT and RLHF methods for preference alignment. The work explores fine tuning on multi-GPU environment utilizing distributed training methods like DeepSpeed and FSDP using the ```accelerate``` library provided by HuggingFace.


### Stack

* LLM - ```Meta-Llama-3-8B```
* Dataset (HF) - ```mlabonne/orpo-dpo-mix-40k```
* Fine-Tuning Method - ```ORPO```
* Accelerator Technique - ```DeepSpeed ZeRO-3```
* Trainer API - ```HuggingFace```
* Run-time environment - ```multi-GPU (2x NVIDIA RTX 6000)```

### How To Use
1. Create conda environment using ```llama.yml```
```
conda env create -f llama.yml
```

2. Run ```llm_llama3_fine_tuning_orpo.ipynb```

### Caution
- Put the token issued by Hugging Face into the ```HF_TOKEN``` variable.
- Before you start ```notebook_launcher(main, num_processes=2)``` cell, the result of ```torch.cuda.is_initialized()
``` must be ```False```. If the result is True, The cell returns [Error](https://github.com/huggingface/accelerate/issues/940).


### Acknowledgments

Thanks for the work shared by Maxime Labonn in his blog [here](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html).
