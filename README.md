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



首先感谢Chatgpt与Claude对本项目的大力支持。

我使用的是Ubuntu22.04，系统配置如下
<img width="2162" height="1057" alt="image" src="https://github.com/user-attachments/assets/1715a059-9270-44ed-b57f-0f1b730856fe" />
最核心的显卡是英伟达的3090，24G的显存可以说是卡在模型训练的最低要求，并不是图便宜，而是我选的那家服务器提供商大多的配置已经被抢购空了，只剩下特别贵的与特别捞的，恰好剩下一台3090刚好能满足要求。

具体教程在原项目已经给的很清楚了，照着做就行了。
即便如此，我在此期间也踩了不少的坑。首先就是Python与Python3的不同，显示的command not found让我一度以为这台机子没装Python，后来才发现它预装的是Python3，而早期的Linux默认的输入python命令是调用Python2.x，所以很多系统都是保留两个命令用于兼容旧程序。
那还能说啥了，把代码里的python改python3呗。

Python解决了就开始环境准备，在测试Torch是否可用cuda时我再次犯了个错，直接把print命令输到linux根目录里了，结果当然又是喜闻乐见的command not found。各位初学者在这里一定要记得先打开Python3，在>>>后面输入那条检测命令。

在开始训练前我花了不少时间去准备。服务器git clone的速度简直像是龟爬，我又不会在linux上设置网络代理，好在数据总量并不算大，忍一忍也就过去了。在下载训练用的数据时我直接求助claude，而在那之前gpt已经被我问到上限了。好在挂载训练数据的网站没有限制速度，这一步也很快完成。


预训练这一步花了我大约一个小时。不得不说大佬的教程就是NB，一个小时跑下来一点问题都没出现。


然后就是监督微调，我没想到原来这一步才是大头。整整两轮共四个小时，甚至我写到这里时还差两个小时才能训练完。
还好3090的服务器比较便宜。
<img width="2174" height="1037" alt="image" src="https://github.com/user-attachments/assets/699ab289-a01d-44ff-9c41-fdd8b56dfd7e" />
不过我要是换更高配置也许根本不用花这么多时间，具体的耗费我会留到最后一起总结。



<img width="2155" height="1023" alt="image" src="https://github.com/user-attachments/assets/005ad71d-a382-4aea-b405-3806d6152b1f" />
跑是能跑，但和人机没啥区别。
我给它的训练数据太少了，虽然它达到了99M的大小，但依然智能低下。
<img width="2160" height="596" alt="image" src="https://github.com/user-attachments/assets/c8bdda44-d5aa-4928-80ee-3c845d8d1afe" />
它自杀了
