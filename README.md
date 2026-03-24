# 基于Minimind项目训练的小模型
这是我的第一个Github项目，我正在努力
<img width="2405" height="586" alt="logo" src="https://github.com/user-attachments/assets/97f1bfa0-1777-4ce3-9766-5a37f7d600be" />
<div align="center">
</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/12586)

</div>

<div align="center">
  <h3>"大道至简"</h3>
</div>

<div align="center">



* 此开源项目旨在完全从0开始，仅用3块钱成本 + 2小时！即可训练出仅为25.8M的超小语言模型**MiniMind**。
* **MiniMind**系列极其轻量，最小版本体积是 GPT-3 的 $\frac{1}{7000}$，力求做到最普通的个人GPU也可快速训练。
* 项目同时开源了大模型的极简结构-包含拓展共享混合专家(MoE)、数据集清洗、预训练(Pretrain)、监督微调(SFT)、LoRA微调、直接偏好优化(DPO)、强化学习训练(RLAIF: PPO/GRPO等)、模型蒸馏等全过程代码。
* **MiniMind**同时拓展了视觉多模态的VLM: [MiniMind-V](https://github.com/jingyaogong/minimind-v)。
* 项目所有核心算法代码均从0使用PyTorch原生重构！不依赖第三方库提供的抽象接口。
* 这不仅是大语言模型的全阶段开源复现，也是一个入门LLM的教程。
* 希望此项目能为所有人提供一个抛砖引玉的示例，一起感受创造的乐趣！推动更广泛AI社区的进步！




<div align="center">

![minimind2](./images/minimind2.gif)

[🔗🍓推理模型](https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning) | [🔗🤖常规模型](https://www.modelscope.cn/studios/gongjy/MiniMind) | [🔗🎞️视频介绍](https://www.bilibili.com/video/BV12dHPeqE72/?share_source=copy_web&vd_source=670c2504f88726f8cf4a21ef6147c0e8)


<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5" style="text-decoration: none;">
          <img src="./images/and_huggingface.png" alt="Hugging Face Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
      <td align="center">
        <a href="https://www.modelscope.cn/profile/gongjy" style="text-decoration: none;">
          <img src="./images/and_modelscope.png" alt="ModelScope Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
    </tr>
  </table>
</div>


</div>

# 📌 Introduction

大语言模型（Large Language Model, LLM）的出现引发了全世界对AI的空前关注。
无论是ChatGPT、DeepSeek还是Qwen，都以其惊艳的效果令人叹为观止。
然而，动辄数百亿参数的庞大规模，使得它们对个人设备而言不仅难以训练，甚至连部署都显得遥不可及。
打开大模型的“黑盒子”，探索其内部运作机制，多么令人心潮澎湃！
遗憾的是，99%的探索只能止步于使用LoRA等技术对现有大模型进行少量微调，学习一些新指令或任务。
这就好比教牛顿如何使用21世纪的智能手机——虽然有趣，却完全偏离了理解物理本质的初衷。
与此同时，第三方的大模型框架和工具库，如transformers+trl，几乎只暴露了高度抽象的接口。
通过短短10行代码，就能完成“加载模型+加载数据集+推理+强化学习”的全流程训练。
这种高效的封装固然便利，但也像一架高速飞船，将开发者与底层实现隔离开来，阻碍了深入探究LLM核心代码的机会。
然而，“用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！”。
更糟糕的是，互联网上充斥着大量付费课程和营销号，以漏洞百出、一知半解的内容推销AI教程。
正因如此，本项目初衷是拉低LLM的学习门槛，让每个人都能从理解每一行代码开始，
从零开始亲手训练一个极小的语言模型。是的，从**零开始训练**，而不是仅仅进行**推理**！
最低只需3块钱不到的服务器成本，就能亲身体验从0到1构建一个语言模型的全过程。
一起感受创造的乐趣吧！

> [!NOTE]
> （截至2025-10）MiniMind系列已完成多个型号模型的预训练，最小仅需25.8M（0.02B），即可具备流畅对话能力！

<details style="color:rgb(128,128,128)">
<summary>Models List</summary>

| 模型 (大小)                 | 推理占用 (约) | Release    | 
|-------------------------|----------|------------|
| MiniMind2-small (26M)   | 0.5 GB   | 2025.04.26 |
| MiniMind2-MoE (145M)    | 1.0 GB   | 2025.04.26 |
| MiniMind2 (104M)        | 1.0 GB   | 2025.04.26 |
| minimind-v1-small (26M) | 0.5 GB   | 2024.08.28 |
| minimind-v1-moe (4×26M) | 1.0 GB   | 2024.09.17 |
| minimind-v1 (108M)      | 1.0 GB   | 2024.09.01 |

</details>

**项目包含**

- MiniMind-LLM结构的全部代码（Dense+MoE模型）。
- 包含Tokenizer分词器详细训练代码。
- 包含Pretrain、SFT、LoRA、RLHF-DPO、RLAIF(PPO/GRPO/SPO)、模型蒸馏的全过程训练代码。
- 收集、蒸馏、整理并清洗去重所有阶段的高质量数据集，且全部开源。
- 从0实现预训练、指令微调、LoRA、DPO/PPO/GRPO/SPO强化学习，白盒模型蒸馏。关键算法几乎不依赖第三方封装的框架，且全部开源。
- 同时兼容`transformers`、`trl`、`peft`等第三方主流框架。
- 训练支持单机单卡、单机多卡(DDP、DeepSpeed)训练，支持wandb/swanlab可视化训练流程。支持动态启停训练。
- 在第三方测评榜（C-Eval、C-MMLU、OpenBookQA等）进行模型测试，支持YaRN算法执行RoPE长文本外推。
- 实现Openai-Api协议的极简服务端，便于集成到第三方ChatUI使用（FastGPT、Open-WebUI等）。
- 基于streamlit实现最简聊天WebUI前端。
- 全面兼容社区热门`llama.cpp`、`vllm`、`ollama`推理引擎或`Llama-Factory`训练框架。
- 复现(蒸馏/RL)大型推理模型DeepSeek-R1的MiniMind-Reason模型，**数据+模型**全部开源！

希望此开源项目可以帮助LLM初学者快速入门！

### 👉**更新日志**

<details close> 
<summary> <b>2025-10-24</b> </summary>

- 🔥 新增RLAIF训练算法：PPO、GRPO、SPO（从0原生实现）
- 新增断点续训功能：支持训练自动恢复、跨GPU数量恢复、wandb记录连续性
- 新增RLAIF数据集：rlaif-mini.jsonl（从SFT数据随机采样1万条）；简化DPO数据集，加入中文数据
- 新增YaRN算法：支持RoPE长文本外推，提升长序列处理能力
- Adaptive Thinking：Reason模型可选是否启用思考链
- chat_template全面支持Tool Calling和Reasoning标签（`<tool_call>`、`<think>`等）
- 新增RLAIF完整章节、训练曲线对比、算法原理折叠说明
- [SwanLab](https://swanlab.cn/)替代WandB（国内访问友好，API完全兼容）
- 规范化所有代码 & 修复一些已知bugs

</details>

<details close> 
<summary> <b>2025-04-26</b> </summary>

- 重要更新
- 如有兼容性需要，可访问[🔗旧仓库内容🔗](https://github.com/jingyaogong/minimind/tree/7da201a944a90ed49daef8a0265c959288dff83a)。
- MiniMind模型参数完全改名，对齐Transformers库模型（统一命名）。
- generate方式重构，继承自GenerationMixin类。
- 🔥支持llama.cpp、vllm、ollama等热门三方生态。
- 规范代码和目录结构。
- 改动词表`<s></s>`->`<|im_start|><|im_end|>`

```text
为兼容第三方推理框架llama.cpp、vllm，本次更新需付出一些可观代价。
本次更新不再支持「直接」加载25-04-26以前的旧模型进行推理。
由于Llama位置编码方式与minimind存在区别，导致映射Llama模型后QK值存在差异
MiniMind2系列旧模型均经过权重映射+（微调训练）QKVO线性层校准恢复而来。
本次更新后将放弃对`minimind-v1`全系列的维护，并在仓库中下线。
```

</details>

<details close> 
<summary> <b>2025-02-09</b> </summary>

- 迎来发布以来重大更新，Release MiniMind2 Series。
- 代码几乎全部重构，使用更简洁明了的统一结构。
  如有旧代码的兼容性需要，可访问[🔗旧仓库内容🔗](https://github.com/jingyaogong/minimind/tree/6e9cd28ef9b34a0a10afbdf6f59e65cb6e628efb)。
- 免去数据预处理步骤。统一数据集格式，更换为`jsonl`格式杜绝数据集下载混乱的问题。
- MiniMind2系列效果相比MiniMind-V1显著提升。
- 小问题：{kv-cache写法更标准、MoE的负载均衡loss被考虑等等}
- 提供模型迁移到私有数据集的训练方案（医疗模型、自我认知样例）。
- 精简预训练数据集，并大幅提升预训练数据质量，大幅缩短个人快速训练所需时间，单卡3090即可2小时复现！
- 更新：LoRA微调脱离peft包装，从0实现LoRA过程；DPO算法从0使用PyTorch原生实现；模型白盒蒸馏原生实现。
- MiniMind2-DeepSeek-R1系列蒸馏模型诞生！
- MiniMind2具备一定的英文能力！
- 更新MiniMind2与第三方模型的基于更多大模型榜单测试性能的结果。

</details>

<details close>
<summary> <b>More...</b> </summary>

**2024-10-05**
- 为MiniMind拓展了多模态能力之---视觉
- 移步孪生项目[minimind-v](https://github.com/jingyaogong/minimind-v)查看详情！

**2024-09-27**
- 09-27更新pretrain数据集的预处理方式，为了保证文本完整性，放弃预处理成.bin训练的形式（轻微牺牲训练速度）。
- 目前pretrain预处理后的文件命名为：pretrain_data.csv。
- 删除了一些冗余的代码。

**2024-09-17**
- 更新minimind-v1-moe模型
- 为了防止歧义，不再使用mistral_tokenizer分词，全部采用自定义的minimind_tokenizer作为分词器。

**2024-09-01**
- 更新minimind-v1 (108M)模型，采用minimind_tokenizer，预训练轮次3 + SFT轮次10，更充分训练，性能更强。
- 项目已部署至ModelScope创空间，可以在此网站上体验：
- [🔗ModelScope在线体验🔗](https://www.modelscope.cn/studios/gongjy/minimind)

**2024-08-27**
- 项目首次开源

</details>

# 📌 快速开始

<details style="color:rgb(128,128,128)">
<summary>分享本人的软硬件配置（仅供参考）</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

### 第0步

```bash
git clone https://github.com/jingyaogong/minimind.git
```

## Ⅰ 测试已有模型效果

### 1.环境准备

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 2.下载模型

到项目根目录

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2 # or https://www.modelscope.cn/models/gongjy/MiniMind2
```

### （可选）命令行问答

```bash
# 使用transformers格式模型
python eval_llm.py --load_from ./MiniMind2
```

### （可选）启动WebUI

```bash
# 可能需要`python>=3.10` 安装 `pip install streamlit`
# cd scripts
streamlit run web_demo.py
```

### （可选）第三方推理框架

```bash
# ollama
ollama run jingyaogong/minimind2
# vllm
vllm serve ./MiniMind2/ --served-model-name "minimind"
```

## Ⅱ 从0开始自己训练

### 1.环境准备

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

<details style="color:rgb(128,128,128)">
<summary>注：提前测试Torch是否可用cuda</summary>

```bash
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下载whl文件安装。参考[链接](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2.数据下载

从下文提供的[数据集下载链接](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
下载需要的数据文件（创建`./dataset`目录）并放到`./dataset`下

<details style="color:rgb(128,128,128)">
<summary>注：数据集须知</summary>

默认推荐下载`pretrain_hq.jsonl` + `sft_mini_512.jsonl`最快速度复现Zero聊天模型。

数据文件可自由选择，下文提供了多种搭配方案，可根据自己手头的训练需求和GPU资源进行适当组合。

</details>

### 3.开始训练

目录位于`trainer`

<details style="color:rgb(128,128,128)">
<summary>💡 检查点暂停续训</summary>

所有训练脚本均自动保存检查点，只需添加 `--from_resume 1` 参数即可自动检测加载&恢复训练：

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
...
```

**断点续训机制说明：**
- 训练过程自动在 `./checkpoints/` 目录保存完整检查点（模型、优化器、训练进度等）
- 检查点文件命名：`<权重名>_<维度>_resume.pth`（如：`full_sft_512_resume.pth`）
- 支持跨不同GPU数量恢复（自动调整step）
- 支持wandb训练记录连续性（自动恢复同一个run）

> 适合长时间训练或不稳定环境，无需担心训练中断导致进度丢失

</details>

**3.1 预训练（学知识）**

```bash
python train_pretrain.py
```

> 执行预训练，得到 `pretrain_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为512）


**3.2 监督微调（学对话方式）**

```bash
python train_full_sft.py
```

> 执行监督微调，得到 `full_sft_*.pth` 作为指令微调的输出权重（其中`full`即为全参数微调）

<details style="color:rgb(128,128,128)">
<summary>注：训练须知</summary>

所有训练过程默认每隔100步保存1次参数到文件`./out/***.pth`（每次会覆盖掉旧权重文件）。

简单起见，此处只写明两个阶段训练过程。如需其它训练 (LoRA, 蒸馏, 强化学习, 微调推理等) 可参考下文【实验】小节的详细说明。

</details>


---

### 4.测试自己训练的模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。
也可以直接去[此处](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/files)下载使用我训练的`*.pth`文件。

```bash
python eval_llm.py --weight full_sft # 或 pretrain/dpo/ppo/grpo...
```

<details style="color:rgb(128,128,128)">
<summary>注：测试须知</summary>

`--weight` 参数指定权重名称前缀，可选：`pretrain`, `full_sft`, `dpo`, `reason`, `ppo_actor`, `grpo`, `spo` 等

其他常用参数：
- `--load_from`: 模型加载路径（`model`=原生torch权重，其他路径=transformers格式）
- `--save_dir`: 模型权重目录（默认`out`）
- `--lora_weight`: LoRA权重名称（`None`表示不使用）
- `--historys`: 携带历史对话轮数（需为偶数，0表示不携带历史）
- `--max_new_tokens`: 最大生成长度（默认8192）
- `--temperature`: 生成温度（默认0.85）
- `--top_p`: nucleus采样阈值（默认0.85）


使用方式直接查看`eval_llm.py`代码即可。

</details>


---

> [!TIP]
> 所有训练脚本均为Pytorch原生框架，均支持多卡加速，假设你的设备有N (N＞1) 张显卡：

单机N卡启动训练方式 (DDP, 支持多机多卡集群)

```bash
torchrun --nproc_per_node N train_xxx.py
```

<details style="color:rgb(128,128,128)">
<summary>注：其它须知</summary>

<del>
单机N卡启动训练 (DeepSpeed)

```bash
deepspeed --master_port 29500 --num_gpus=N train_xxx.py
```
</del>

可根据需要开启wandb记录训练过程（需可直连）

```bash
# 需要登录: wandb login
torchrun --nproc_per_node N train_xxx.py --use_wandb
# and
python train_xxx.py --use_wandb
```

通过添加`--use_wandb`参数，可以记录训练过程，训练完成后，可以在wandb网站上查看训练过程。通过修改`wandb_project`
和`wandb_run_name`参数，可以指定项目名称和运行名称。

【注】：25年6月后，国内网络环境无法直连WandB，MiniMind项目默认转为使用[SwanLab](https://swanlab.cn/)作为训练可视化工具（完全兼容WandB API），即`import wandb`改为`import swanlab as wandb`即可，其他均无需改动。

</details>

# 📌 数据介绍

## Ⅰ Tokenizer

分词器将单词从自然语言通过“词典”映射到`0, 1, 36`这样的数字，可以理解为数字就代表了单词在“词典”中的页码。
可以选择自己构造词表训练一个“词典”，代码可见`./trainer/train_tokenizer.py`（仅供学习参考，若非必要无需再自行训练，MiniMind已自带tokenizer）。
或者选择比较出名的开源大模型分词器，
正如同直接用新华/牛津词典的优点是token编码压缩率很好，缺点是页数太多，动辄数十万个词汇短语；
自己训练的分词器，优点是词表长度和内容随意控制，缺点是压缩率很低（例如"hello"也许会被拆分为"h e l l o"
五个独立的token），且生僻词难以覆盖。
“词典”的选择固然很重要，LLM的输出本质上是SoftMax到词典N个词的多分类问题，然后通过“词典”解码到自然语言。
因为MiniMind体积需要严格控制，为了避免模型头重脚轻（词嵌入embedding层参数在LLM占比太高），所以词表长度短短益善。

<details style="color:rgb(128,128,128)">
<summary>Tokenizer介绍</summary>

第三方强大的开源模型例如Yi、qwen、chatglm、mistral、Llama3的tokenizer词表长度如下：

<table>
  <tr><th>Tokenizer模型</th><th>词表大小</th><th>来源</th></tr>
  <tr><td>yi tokenizer</td><td>64,000</td><td>01万物（中国）</td></tr>
  <tr><td>qwen2 tokenizer</td><td>151,643</td><td>阿里云（中国）</td></tr>
  <tr><td>glm tokenizer</td><td>151,329</td><td>智谱AI（中国）</td></tr>
  <tr><td>mistral tokenizer</td><td>32,000</td><td>Mistral AI（法国）</td></tr>
  <tr><td>llama3 tokenizer</td><td>128,000</td><td>Meta（美国）</td></tr>
  <tr><td>minimind tokenizer</td><td>6,400</td><td>自定义</td></tr>
</table>

> 👉2024-09-17更新：为了防止过去的版本歧义&控制体积，minimind所有模型均使用minimind_tokenizer分词，废弃所有mistral_tokenizer版本。

```
# 一些自言自语
> 尽管minimind_tokenizer长度很小，编解码效率弱于qwen2、glm等中文友好型分词器。
> 但minimind模型选择了自己训练的minimind_tokenizer作为分词器，以保持整体参数轻量，避免编码层和计算层占比失衡，头重脚轻，因为minimind的词表大小只有6400。
> 且minimind在实际测试中没有出现过生僻词汇解码失败的情况，效果良好。
> 由于自定义词表压缩长度到6400，使得LLM总参数量最低只有25.8M。
> 训练数据`pretrain_hq.jsonl`均来自于`匠数大模型数据集`，这部分数据相对次要，如需训练可以自由选择。
```

</details>

## Ⅱ Pretrain数据

经历了MiniMind-V1的低质量预训练数据，导致模型胡言乱语的教训，`2025-02-05` 之后决定不再采用大规模无监督的数据集做预训练。
进而尝试把[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)的中文部分提取出来，
清洗出字符`<512`长度的大约1.6GB的语料直接拼接成预训练数据 `pretrain_hq.jsonl`，hq即为high
quality（当然也还不算high，提升数据质量无止尽）。

文件`pretrain_hq.jsonl` 数据格式为

```json
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

## Ⅲ SFT数据

[匠数大模型SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
“是一个完整、格式统一、安全的大模型训练和研究资源。
从网络上的公开数据源收集并整理了大量开源数据集，对其进行了格式统一，数据清洗，
包含10M条数据的中文数据集和包含2M条数据的英文数据集。”
以上是官方介绍，下载文件后的数据总量大约在4B tokens，肯定是适合作为中文大语言模型的SFT数据的。
但是官方提供的数据格式很乱，全部用来sft代价太大。
我将把官方数据集进行了二次清洗，把含有符号污染和噪声的条目去除；另外依然只保留了总长度`<512`
的内容，此阶段希望通过大量对话补充预训练阶段欠缺的知识。
导出文件为`sft_512.jsonl`(~7.5GB)。

[Magpie-SFT数据集](https://www.modelscope.cn/organization/Magpie-Align)
收集了~1M条来自Qwen2/2.5的高质量对话，我将这部分数据进一步清洗，把总长度`<2048`的部分导出为`sft_2048.jsonl`(~9GB)。
长度`<1024`的部分导出为`sft_1024.jsonl`(~5.5GB)，用大模型对话数据直接进行sft就属于“黑盒蒸馏”的范畴。

进一步清洗前两步sft的数据（只保留中文字符占比高的内容），筛选长度`<512`的对话，得到`sft_mini_512.jsonl`(~1.2GB)。

所有sft文件 `sft_X.jsonl` 数据格式均为

```text
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

## Ⅳ RLHF数据

来自[Magpie-DPO数据集](https://www.modelscope.cn/datasets/Magpie-Align/MagpieLM-DPO-Data-v0.1)
大约200k条偏好数据（均是英文）生成自Llama3.1-70B/8B，可以用于训练奖励模型，优化模型回复质量，使其更加符合人类偏好。
这里将数据总长度`<3000`的内容重组为`dpo.jsonl`(~0.9GB)，包含`chosen`和`rejected`两个字段，`chosen`
为偏好的回复，`rejected`为拒绝的回复。

文件 `dpo.jsonl` 数据格式为

```text
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

## Ⅴ Reason数据集：

不得不说2025年2月谁能火的过DeepSeek...
也激发了我对RL引导的推理模型的浓厚兴趣，目前已经用Qwen2.5复现了R1-Zero。
如果有时间+效果work（但99%基模能力不足）我会在之后更新MiniMind基于RL训练的推理模型而不是蒸馏模型。
时间有限，最快的低成本方案依然是直接蒸馏（黑盒方式）。
耐不住R1太火，短短几天就已经存在一些R1的蒸馏数据集[R1-Llama-70B](https://www.modelscope.cn/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)、[R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)、
[Alpaca-Distill-R1](https://huggingface.co/datasets/shareAI/Alpaca-Distill-R1-ZH)、
[deepseek_r1_zh](https://huggingface.co/datasets/jinliuxi/deepseek_r1_zh)等等，纯中文的数据可能比较少。
最终整合它们，导出文件为`r1_mix_1024.jsonl`，数据格式和`sft_X.jsonl`一致。

## Ⅵ 更多数据集

目前已经有[HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)
在收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料，并持续更新这方面的最新进展。全面且专业，Respect！

---

## Ⅷ MiniMind训练数据集

> [!NOTE]
> 2025-02-05后，开源MiniMind最终训练所用的所有数据集，因此无需再自行预处理大规模数据集，避免重复性的数据处理工作。

MiniMind训练数据集下载地址： [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

> 无需全部clone，可单独下载所需的文件

将下载的数据集文件放到`./dataset/`目录下（✨为推荐的必须项）

```bash
./dataset/
├── dpo.jsonl (55MB, ✨)
├── lora_identity.jsonl (22.8KB)
├── lora_medical.jsonl (34MB)
├── pretrain_hq.jsonl (1.6GB, ✨)
├── r1_mix_1024.jsonl (340MB)
├── rlaif-mini.jsonl (1MB, ✨)
├── sft_1024.jsonl (5.6GB)
├── sft_2048.jsonl (9GB)
├── sft_512.jsonl (7.5GB)
└── sft_mini_512.jsonl (1.2GB, ✨)
```

<details style="color:rgb(128,128,128)">
<summary>注：各数据集简介</summary>

* `dpo.jsonl`✨ --RLHF阶段数据集（已精简优化，适合快速训练）
* `lora_identity.jsonl` --自我认知数据集（例如：你是谁？我是minimind...），推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `lora_medical.jsonl` --医疗问答数据集，推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `pretrain_hq.jsonl`✨ --预训练数据集，整合自匠数科技（推荐设置`max_seq_len≈320`）
* `r1_mix_1024.jsonl` --DeepSeek-R1-1.5B蒸馏数据，每条数据字符最大长度为1024（推荐设置`max_seq_len≈720`）
* `rlaif-mini.jsonl` --RLAIF训练数据集，从SFT数据集中随机采样1万条高质量对话，用于PPO/GRPO/SPO等强化学习算法训练
* `sft_1024.jsonl` --整合自Qwen2.5蒸馏数据（是sft_2048的子集），每条数据字符最大长度为1024（推荐设置`max_seq_len≈650`）
* `sft_2048.jsonl` --整合自Qwen2.5蒸馏数据，每条数据字符最大长度为2048（推荐设置`max_seq_len≈1400`）
* `sft_512.jsonl` --整合自匠数科技SFT数据，每条数据字符最大长度为512（推荐设置`max_seq_len≈350`）
* `sft_mini_512.jsonl`✨ --极简整合自匠数科技SFT数据+Qwen2.5蒸馏数据（用于快速训练Zero模型），每条数据字符最大长度为512（推荐设置`max_seq_len≈340`）


训练参数`max_seq_len`目前指的是tokens长度，而非绝对字符数。
本项目tokenizer在中文文本上大约`1.5~1.7 字符/token`，纯英文的压缩比在`4~5 字符/token`，不同数据分布会有波动。
数据集命名标注的“最大长度”均为字符数，100长度的字符串可粗略换算成`100/1.5≈67`的tokens长度。

例如：

* 中文：`白日依山尽`5个字符可能被拆分为[`白日`,`依`,`山`,`尽`] 4个tokens；
* 英文：`The sun sets in the west`24个字符可能被拆分为[`The `,`sun `,`sets `,`in `,`the`,`west`] 6个tokens

“推荐设置”给出了各个数据集上最大tokens长度的粗略估计。
须知max_seq_len可以激进/保守/均衡地调整，因为更大或更小均无法避免副作用：一些样本短于max_seq_len后被padding浪费算力，一些样本长于max_seq_len后被截断语意。

在 `算力效率` <---> `语义完整性` 之间找到一个平衡点即可

</details>


![dataset](./images/dataset.jpg)

<details style="color:rgb(128,128,128)">
<summary>说明 & 推荐训练方案</summary>

* MiniMind2 Series均经过共约20GB语料训练，大约4B tokens，即对应上面的数据组合训练结果（开销：💰💰💰💰💰💰💰💰，效果：😊😊😊😊😊😊）

* 想要最快速度从0实现Zero模型，推荐使用`pretrain_hq.jsonl` + `sft_mini_512.jsonl` 的数据组合，具体花销和效果可查看下文表格（开销：💰，效果：😊😊）

* 推荐具备一定算力资源或更在意效果的朋友可以考虑前者完整复现MiniMind2；仅有单卡GPU或在乎短时间快速复现的朋友强烈推荐后者；

* 【折中方案】亦可选择例如`sft_mini_512.jsonl`、`sft_1024.jsonl`中等规模数据进行自由组合训练（开销：💰💰💰，效果：😊😊😊😊）。

</details>

# 📌 Model

## Structure

MiniMind-Dense（和[Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)一样）使用了Transformer的Decoder-Only结构，跟GPT-3的区别在于：

* 采用了GPT-3的预标准化方法，也就是在每个Transformer子层的输入上进行归一化，而不是在输出上。具体来说，使用的是RMSNorm归一化函数。
* 用SwiGLU激活函数替代了ReLU，这样做是为了提高性能。
* 像GPT-Neo一样，去掉了绝对位置嵌入，改用了旋转位置嵌入（RoPE），这样在处理超出训练长度的推理时效果更好。

---

MiniMind-MoE模型，它的结构基于Llama3和[Deepseek-V2/3](https://arxiv.org/pdf/2405.04434)中的MixFFN混合专家模块。

* DeepSeek-V2在前馈网络（FFN）方面，采用了更细粒度的专家分割和共享的专家隔离技术，以提高Experts的效果。

---

MiniMind的整体结构一致，只是在RoPE计算、推理函数和FFN层的代码上做了一些小调整。
其结构如下图（重绘版）：

![structure](./images/LLM-structure.png)
![structure-moe](./images/LLM-structure-moe.png)

修改模型配置见[./model/model_minimind.py](./model/model_minimind.py)。
参考模型参数版本见下表：

| Model Name        | params | len_vocab | rope_theta | n_layers | d_model | kv_heads | q_heads | share+route |
|-------------------|--------|-----------|------------|----------|---------|----------|---------|-------------|
| MiniMind2-Small   | 26M    | 6400      | 1e6        | 8        | 512     | 2        | 8       | -           |
| MiniMind2-MoE     | 145M   | 6400      | 1e6        | 8        | 640     | 2        | 8       | 1+4         |
| MiniMind2         | 104M   | 6400      | 1e6        | 16       | 768     | 2        | 8       | -           |
| minimind-v1-small | 26M    | 6400      | 1e4        | 8        | 512     | 8        | 16      | -           |
| minimind-v1-moe   | 4×26M  | 6400      | 1e4        | 8        | 512     | 8        | 16      | 1+4         |
| minimind-v1       | 108M   | 6400      | 1e4        | 16       | 768     | 8        | 16      | -           |


## Model Configuration

📋关于LLM的参数配置，有一篇很有意思的论文[MobileLLM](https://arxiv.org/pdf/2402.14905)做了详细的研究和实验。
Scaling Law在小模型中有自己独特的规律。
引起Transformer参数成规模变化的参数几乎只取决于`d_model`和`n_layers`。

* `d_model`↑ + `n_layers`↓ -> 矮胖子
* `d_model`↓ + `n_layers`↑ -> 瘦高个

2020年提出Scaling Law的论文认为，训练数据量、参数量以及训练迭代次数才是决定性能的关键因素，而模型架构的影响几乎可以忽视。
然而似乎这个定律对小模型并不完全适用。
MobileLLM提出架构的深度比宽度更重要，「深而窄」的「瘦长」模型可以学习到比「宽而浅」模型更多的抽象概念。
例如当模型参数固定在125M或者350M时，30～42层的「狭长」模型明显比12层左右的「矮胖」模型有更优越的性能，
在常识推理、问答、阅读理解等8个基准测试上都有类似的趋势。
这其实是非常有趣的发现，因为以往为100M左右量级的小模型设计架构时，几乎没人尝试过叠加超过12层。
这与MiniMind在训练过程中，模型参数量在`d_model`和`n_layers`之间进行调整实验观察到的效果是一致的。
然而「深而窄」的「窄」也是有维度极限的，当d_model<512时，词嵌入维度坍塌的劣势非常明显，
增加的layers并不能弥补词嵌入在固定q_head带来d_head不足的劣势。
当d_model>1536时，layers的增加似乎比d_model的优先级更高，更能带来具有"性价比"的参数->效果增益。

* 因此MiniMind设定small模型dim=512，n_layers=8来获取的「极小体积<->更好效果」的平衡。
* 设定dim=768，n_layers=16来获取效果的更大收益，更加符合小模型Scaling-Law的变化曲线。

作为参考，GPT3的参数设定见下表：
![gpt3_config.png](./images/gpt3_config.png)

---

# 📌 Experiment

## Ⅰ 训练开销

- **时间单位**：小时 (h)。
- **成本单位**：人民币 (￥)；7￥ ≈ 1美元。
- **3090 租卡单价**：≈1.3￥/h（可自行参考实时市价）。
- **参考标准**：表格仅实测 `pretrain` 和 `sft_mini_512` 两个数据集的训练时间，其它耗时根据数据集大小估算（可能存在些许出入）。

> 基于 3090 （单卡）成本计算

| Model Name      | params | pretrain         | sft_mini_512     | sft_512       | sft_1024          | sft_2048         | RLHF          |
|-----------------|--------|------------------|------------------|---------------|-------------------|------------------|---------------|
| MiniMind2-Small | 26M    | ≈1.1h<br/>≈1.43￥ | ≈1h<br/>≈1.3￥    | ≈6h<br/>≈7.8￥ | ≈4.58h<br/>≈5.95￥ | ≈7.5h<br/>≈9.75￥ | ≈1h<br/>≈1.3￥ |
| MiniMind2       | 104M   | ≈3.9h<br/>≈5.07￥ | ≈3.3h<br/>≈4.29￥ | ≈20h<br/>≈26￥ | ≈15h<br/>≈19.5￥   | ≈25h<br/>≈32.5￥  | ≈3h<br/>≈3.9￥ |

---

<details style="color:rgb(128,128,128)">
<summary>训练开销总结&预测</summary>


> MiniMind2-Small参数
>> `pretrain_hq`+`sft_mini_512`数据集
<br/>单卡3090 (1 epoch) + 2.1小时 + 花费2.73元人民币
<br/>即可从0训练出MiniMind-Zero-0.025B模型!!!

> MiniMind2-Small参数
>> `pretrain_hq`+`sft_512`+`sft_2048`+`dpo`数据集
<br/>单卡3090 (2 epochs) + 大约38.16小时 + 花费49.61元人民币
<br/>即可从0训练出MiniMind2-Small-0.025B模型!!!

> MiniMind2参数
>> `pretrain_hq`+`sft_512`+`sft_2048`+`dpo`数据集
<br/>单卡3090 (2 epochs) + 大约122小时 + 花费158.6元人民币
<br/>即可从0训练出MiniMind2-0.1B模型!!!

</details>



✨基于单卡NVIDIA 3090的`MiniMind-Zero`从0训练仅需`2小时` + `3块钱`，实现ChatBot效果！

✨PS：若采用8卡4090训练，总用时甚至可以压缩到10分钟以内！（由于时间更短，花费同样3元左右，与单卡成本相当）

✨以极低极低的门槛，实现人人可玩的大模型自由！这正是MiniMind系列的诞生初衷！

✨仅价值`3块钱`成本的`MiniMind-Zero`并不是噱头！Chat测试：

```text
👶: 请介绍一下自己。
🤖️: 作为人工智能，我没有实际的生活，也没有自我意识，所以没有自己的生活。我被设计成能够帮助用户解答问题、提供信息、进行对话等。我的设计和功能是由计算机科学、机器学习、人工智能技术和算法所构建的，我通过编程和算法实现。

👶: 推荐一些杭州的特色美食吧。
🤖️: 杭州有很多美味的特色美食，比如鸡肉串、鳗鱼头、水煮鱼、豆腐脑等。这些美食都非常美味，有着独特的风味和口感，绝对是值得一试的美食。

👶: 请为我讲解“大语言模型”这个概念。
🤖️: 大语言模型是一种人工智能技术，它使用大量的文本数据来训练模型，然后在不断的迭代中不断优化模型。它的目标是根据输入的数据来生成符合特定需求的文本，这些文本可以是任何你感兴趣的主题。

👶: Introduce the history of the United States, please.
🤖️: 您提到的“Introok's the believeations of theument." 这个名字来源于中国古代的"groty of of the change."
```

极速且初具效果，甚至仍然可以进一步压缩获取更小更优质的训练数据。
Zero模型权重保存为 `full_sft_512_zero.pth`（见下文MiniMind模型文件链接），如有兴趣可下载检验此模型效果。


---

## Ⅱ 主要训练（必须）

> 所有训练脚本均 `cd ./trainer` 目录执行

### **1. 预训练 (Pretrain)**:

LLM首先要学习的并非直接与人交流，而是让网络参数中充满知识的墨水，“墨水” 理论上喝的越饱越好，产生大量的对世界的知识积累。
预训练就是让Model先埋头苦学大量基本的知识，例如从Wiki百科、新闻、书籍整理大规模的高质量训练数据。
这个过程是“无监督”的，即人类不需要在过程中做任何“有监督”的校正，而是由模型自己从大量文本中总结规律学习知识点。
模型此阶段目的只有一个：**学会词语接龙**。例如输入"秦始皇"四个字，它可以接龙"是中国的第一位皇帝"。

```bash
torchrun --nproc_per_node 1 train_pretrain.py # 1即为单卡训练，可根据硬件情况自行调整 (设置>=2)
# or
python train_pretrain.py
```

> 训练后的模型权重文件默认每隔`100步`保存为: `pretrain_*.pth`（*
> 为模型具体dimension，每次保存时新文件会覆盖旧文件）


| MiniMind2-Small (512dim) | MiniMind2 (768dim) |
|---|---|
| <img src="./images/pre_512_loss.png"> | <img src="./images/pre_768_loss.png"> |

### **2. 有监督微调 (Supervised Fine-Tuning)**:

经过预训练，LLM此时已经掌握了大量知识，然而此时它只会无脑地词语接龙，还不会与人聊天。
SFT阶段就需要把半成品LLM施加一个自定义的聊天模板进行微调。
例如模型遇到这样的模板【问题->回答，问题->回答】后不再无脑接龙，而是意识到这是一段完整的对话结束。
称这个过程为指令微调，就如同让已经学富五车的「牛顿」先生适应21世纪智能手机的聊天习惯，学习屏幕左侧是对方消息，右侧是本人消息这个规律。
在训练时，MiniMind的指令和回答长度被截断在512，是为了节省显存空间。就像学习写作时，会先从短的文章开始，当学会写作200字作文后，800字文章也可以手到擒来。
在需要长度拓展时，只需要准备少量的2k/4k/8k长度对话数据进行进一步微调即可（此时最好配合RoPE-NTK的基准差值）。
> 在推理时通过调整RoPE线性差值，实现免训练长度外推到2048及以上将会很方便。

```bash
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

> 训练后的模型权重文件默认每隔`100步`保存为: `full_sft_*.pth`（*
> 为模型具体dimension，每次保存时新文件会覆盖旧文件）


| MiniMind2-Small (512dim) | MiniMind2 (768dim) |
|---|---|
| <img src="./images/sft_512_loss.png"> | <img src="./images/sft_768_loss.png"> |

## Ⅲ 其它训练阶段（可选）

> 所有训练脚本均 `cd ./trainer` 目录执行

### **3. 知识蒸馏 (Knowledge Distillation, KD)**

在前面的所有训练步骤中，模型已经完全具备了基本能力，通常可以学成出师了。
而知识蒸馏可以进一步优化模型的性能和效率，所谓知识蒸馏，即学生模型面向教师模型学习。
教师模型通常是经过充分训练的大模型，具有较高的准确性和泛化能力。
学生模型是一个较小的模型，目标是学习教师模型的行为，而不是直接从原始数据中学习。
在SFT学习中，模型的目标是拟合词Token分类硬标签（hard labels），即真实的类别标签（如 0 或 6400）。
在知识蒸馏中，教师模型的softmax概率分布被用作软标签（soft labels）。小模型仅学习软标签，并使用KL-Loss来优化模型的参数。
通俗地说，SFT直接学习老师给的解题答案。而KD过程相当于“打开”老师聪明的大脑，尽可能地模仿老师“大脑”思考问题的神经元状态。
例如，当老师模型计算`1+1=2`这个问题的时候，最后一层神经元a状态为0，神经元b状态为100，神经元c状态为-99...
学生模型通过大量数据，学习教师模型大脑内部的运转规律。这个过程即称之为：知识蒸馏。
知识蒸馏的目的只有一个：让小模型体积更小的同时效果更好。
然而随着LLM诞生和发展，模型蒸馏一词被广泛滥用，从而产生了“白盒/黑盒”知识蒸馏两个派别。
GPT-4这种闭源模型，由于无法获取其内部结构，因此只能面向它所输出的数据学习，这个过程称之为黑盒蒸馏，也是大模型时代最普遍的做法。
黑盒蒸馏与SFT过程完全一致，只不过数据是从大模型的输出收集，因此只需要准备数据并且进一步FT即可。
注意更改被加载的基础模型为`full_sft_*.pth`，即基于微调模型做进一步的蒸馏学习。
`./dataset/sft_1024.jsonl`与`./dataset/sft_2048.jsonl` 均收集自qwen2.5-7/72B-Instruct大模型，可直接用于SFT以获取Qwen的部分行为。

```bash
# 注意需要更改train_full_sft.py数据集路径，以及max_seq_len  
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

> 训练后的模型权重文件默认每隔`100步`同样保存为: `full_sft_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）

此处应当着重介绍MiniMind实现的白盒蒸馏代码`train_distillation.py`，由于MiniMind同系列本身并不存在强大的教师模型，因此白盒蒸馏代码仅作为学习参考。

```bash
torchrun --nproc_per_node 1 train_distillation.py
# or
python train_distillation.py
```
<img width="2174" height="1037" alt="image" src="https://github.com/user-attachments/assets/699ab289-a01d-44ff-9c41-fdd8b56dfd7e" />


