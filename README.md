# Contrastive Preference Optimization  论文复现/踩坑记录

本项目旨在复现论文：

> **Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation**  
> 作者：Haoran Xu, Jiayi Wang, Xinyi Wang, Kevin Yang, Yuqing Tang, Ankur Bapna, Orhan Firat  
> 发布于 arXiv:2404.12135 [cs.CL], 2024年4月  
> 论文网址：https://arxiv.org/pdf/2401.08417v2  
> 原仓库： https://github.com/fe1ixxu/ALMA/tree/master

## 项目简介

由于本地设备资源有限，我未从头开始训练模型，而是采用了作者开源的预训练模型 **`ALMA-7B-Pretrain`** 以及其对应的适配器 **`ALMA-7B-Pretrain-LoRA`**。在此基础上，我使用论文中提供的 **`ALMA-R Triplet Preference Data`** 数据集进行了CPO微调，并成功获得了新的适配器权重，读者可自行加载测试效果。权重下载：https://pan.quark.cn/s/4472092b1019?pwd=jNsG

由于算力限制，目前尚未在论文中的官方评测集 **WMT'22** 和 **WMT'23** 上对微调后的模型性能进行完整评估，后续计划视资源情况继续补充。

## 实验环境与硬件配置
| 配置情况       | 规格说明                                              |
|----------------|-------------------------------------------------------|
| **GPU**        | NVIDIA RTX 4090（24GB 显存）×1                        |
| **CPU**        | 16 vCPU Intel(R) Xeon(R) Platinum 8352V @ 2.10GHz     |
| **内存**       | 120 GB                                                |
| **显存**       | 32 GB                                                 |
| **操作系统**   | Ubuntu 22.04                                          |
| **Python 版本**   | 3.10                                               |
| **CUDA 版本**   | 11.8                                                 |

##  项目结构文件说明

| 文件 / 文件夹名                     | 说明 |
|------------------------------------|------|
| `install_alma.sh`                 | 项目初始化与依赖安装脚本 |
| `modeling_xalma.py`              | 自定义模型结构定义 |
| `run_cpo_llmmt.py`               | CPO 微调入口脚本 |
| `run_llmmt.py`                   | 多语言推理与评估脚本 |
| `configs/`                       | DeepSpeed 与训练相关配置文件 |
| `evals/`                         | 多语言评估脚本与测试入口 |
| `human_written_data/`           | 人工翻译数据集（cs-en, zh-en 等语言对） |
| `outputs/`                       | 模型生成的翻译结果，按模型和语言对组织 |
| `runs/`                          | 各阶段训练脚本 |
| `utils/`                         | 工具模块，包括 trainer、数据处理等 |
| `wandb/`                         | wandb 训练过程日志与缓存目录 |

## 复现流程


###  创建并激活虚拟环境

```bash
conda create -n alma python=3.10
conda activate alma
```

###  安装依赖

```bash
bash install_alma.sh
```

该脚本会自动安装所需的 Python 库、transformers、accelerate、peft、deepspeed 等工具。

###  下载预训练模型与适配器权重（以 7B 模型为例）

```bash
git clone https://huggingface.co/haoranxu/ALMA-7B-Pretrain
git clone https://huggingface.co/haoranxu/ALMA-7B-Pretrain-LoRA
```

###  下载 CPO 微调所需三元组偏好数据集

```bash
hf snapshot download haoranxu/ALMA-R-Preference --repo-type dataset --local-dir ./ALMA-R-Preference
```

###  开始微调（执行 CPO 微调训练脚本）

```bash
bash runs/cpo_ft.sh ${your_output_dir}
```

 **注意：**  
请根据你的路径设置，修改 `runs/cpo_ft.sh` 中以下内容：

- `--model_name_or_path`：预训练模型路径  
- `--peft_model_id`：LoRA 适配器路径  
- `--mmt_data_path`：CPO 三元组数据路径  

否则可能会报错找不到模型或数据文件。

###  微调后进行翻译测试

作者已经预处理好测试数据，位于：

```
human_written_data/
```

运行测试脚本：

```bash
bash evals/alma_13b_lora.sh ${your_output_dir}
```

该脚本将加载你训练得到的适配器权重，在多个语言对上进行评估（默认包含 en-de、en-zh、de-en、cs-en 等）。

##  踩坑记录

在复现过程中，我遇到了一些典型错误及环境兼容性问题，以下是详细记录与解决方案：

---

### ❌ `run_cpo_llmmt.py: error: argument --bf16: expected one argument`

- **原因**：`--bf16` 参数未设置值。
- **解决方法**：将 `cpo_ft.sh` 中的 `--bf16` 改为：
  ```bash
  --bf16 True
  ```
- **说明**：`--bf16` 是一个带值参数，必须传入 `True` 或 `False`，否则 argparse 会报错。


---

### ❌ `torch.distributed.DistBackendError: NCCL error: Duplicate GPU detected : rank X and rank 0 both on CUDA device`

- **原因**：多个进程（rank）被错误地分配到了同一个 GPU 上。
- **解决方法**：打开 `configs/deepspeed_train_config_bf16.yaml`，将其中的：
  ```yaml
  num_processes: 8
  ```
  改为：
  ```yaml
  num_processes: 1
  ```

---

### ❌ `ValueError: unknown keys (['use_deepspeed'])`

- **原因**：`use_deepspeed` 参数在 `Trainer` 中未定义。
- **解决方法**：删除 `runs/cpo_ft.sh` 中的以下行：
  ```bash
  --use_deepspeed True
  ```

---

### ❌ `TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'use_flash_attention_2'`

- **原因**：当前模型不接受 `use_flash_attention_2` 参数。
- **解决方法**：打开：
  ```
  utils/utils.py
  ```
  删除第 507 行的：
  ```python
  use_flash_attention_2=True
  ```

---

### ❌ `TypeError: CPOTrainer.__init__() got an unexpected keyword argument 'tokenizer'`

- **原因**：`CPOTrainer` 类中不接收 `tokenizer` 参数。
- **解决方法**：打开：
  ```
  run_cpo_llmmt.py
  ```
  将第 102 行中的：
  ```python
  tokenizer=tokenizer,
  ```
  替换为：
  ```python
  processing_class=processing_class,
  ```

---

### ❌ `ImportError: cannot import name 'is_torch_tpu_available' from 'transformers'`

- **原因**：`transformers >= 4.34` 中已移除该函数，而当前版本受 `trl` 库限制，无法降级。
- **解决方法**：直接删除文件 `run_llmmt.py` 中的以下导入语句：
  ```python
  from transformers import is_torch_tpu_available
  ```

---

### ❌ `ModuleNotFoundError: No module named 'transformers.deepspeed'`

- **原因**：`transformers.deepspeed` 子模块在当前版本中可能已移除或未安装完整。
- **解决方法**：打开：
  ```
  utils/trainer_llmmt.py
  ```
  删除如下导入语句：
  ```python
  from transformers.deepspeed import is_deepspeed_zero3_enabled
  ```

---


