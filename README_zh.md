# MOSS-Audio


<p align="center">
  <img src="./assets/moss-audio-logo.png" width="55%" />
</p>



<div align="center">
  <a href="https://huggingface.co/collections/OpenMOSS-Team/moss-audio"><img src="https://img.shields.io/badge/Huggingface-Models-orange?logo=huggingface&amp"></a>
  <a href="https://modelscope.cn/collections/openmoss/MOSS-Audio"><img src="https://img.shields.io/badge/ModelScope-Models-624AFF?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==&amp"></a>
  <img src="https://img.shields.io/badge/Blog-Coming_Soon-blue?logo=internet-explorer&amp">
  <img src="https://img.shields.io/badge/Arxiv-Coming_Soon-red?logo=Arxiv&amp">

  <a href="https://x.com/Open_MOSS"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
  <a href="https://discord.gg/Xf3aXddCjc"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp"></a>
  <a href="./assets/wechat.png"><img src="https://img.shields.io/badge/WeChat-Join-07C160?logo=wechat&amp;logoColor=white" alt="WeChat"></a>
</div>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>




MOSS-Audio 是由 [MOSI.AI](https://mosi.cn/#hero)、[OpenMOSS 团队](https://www.open-moss.com/) 和 [上海创智学院](https://www.sii.edu.cn/) 推出的开源 **音频理解模型**。它面向复杂真实世界音频进行统一建模，支持 **语音理解、环境声理解、音乐理解、音频描述、时间感知问答和复杂推理**。本次发布共提供 **4 个模型**：**MOSS-Audio-4B-Instruct**、**MOSS-Audio-4B-Thinking**、**MOSS-Audio-8B-Instruct** 和 **MOSS-Audio-8B-Thinking**。其中 Instruct 版本更适合直接指令跟随，Thinking 版本则具备更强的链式思维推理能力。


## 新闻
* 2026.4.13：🎉🎉🎉 我们已发布 [MOSS-Audio](https://huggingface.co/collections/OpenMOSS-Team/moss-audio)。博客和论文即将上线！


## 目录

- [介绍](#introduction)
- [模型架构](#model-architecture)
  - [DeepStack 跨层特征注入](#deepstack-cross-layer-feature-injection)
  - [时间感知表示](#time-aware-representation)
- [已发布模型](#released-models)
- [评测](#evaluation)
- [快速开始](#quickstart)
  - [环境配置](#environment-setup)
  - [基础用法](#basic-usage)
  - [Gradio 应用](#gradio-app)
  - [SGLang 服务](#sglang-serving)
- [更多信息](#more-information)
- [引用](#citation)


## 介绍

<p align="center">
  <img src="./assets/moss-audio-image.png" width="95%" />
</p>



理解音频并不仅仅是将文字转录出来，它还要求模型能够感知声学线索、识别说话人与情绪、理解环境声音、基于时间上下文进行推理，并处理复杂的多步推断。**MOSS-Audio** 的目标就是在单一模型中统一这些能力。

- **语音与内容理解**：准确识别并转录音频输入中的口语内容，生成干净且结构良好的文本输出。支持词级和句级时间戳对齐。
- **说话人、情绪与事件分析**：识别说话人特征，基于语气、音色和上下文分析情绪状态，并检测音频中的关键声学事件。
- **场景与声音线索提取**：从背景声、环境噪声、音乐和非语音信号中提取有效线索，以推断场景上下文与氛围。
- **音乐理解**：分析音乐片段中的风格、情绪变化、配器以及显著声学特征。
- **音频问答与摘要**：针对语音、播客、会议、访谈和环境录音进行问答与摘要生成，帮助用户高效提取关键信息。
- **时间感知问答**：支持时间感知类问题，包括词级和句级时间戳 ASR。
- **复杂推理**：借助链式思维训练与强化学习，对音频内容执行多跳推理。

## 模型架构

<p align="center">
  <img src="./assets/arc2.png" width="95%" />
</p>

MOSS-Audio 采用由三部分组成的模块化设计：音频编码器、模态适配器和大语言模型。原始音频首先由 **MOSS-Audio-Encoder** 编码为 **12.5 Hz** 的连续时序表征，然后通过适配器投影到语言模型的嵌入空间，最终由 LLM 完成自回归文本生成。

我们没有依赖现成的通用音频前端，而是从零训练专用编码器，以获得更鲁棒的语音表征、更紧密的时间对齐能力，以及更好的跨声学域扩展性。


### DeepStack 跨层特征注入

如果仅使用编码器顶层特征，往往会丢失底层韵律、瞬态事件以及局部时频结构。为了解决这一问题，我们在编码器与语言模型之间设计了受 **DeepStack** 启发的跨层注入模块：除了编码器最终层输出外，还会选取更早期和中间层特征，分别进行独立投影，并注入语言模型的前几层，从而保留从低层声学细节到高层语义抽象的多粒度信息。

这一设计尤其适合音频理解任务，因为它有助于保留节奏、音色、瞬态和背景结构，而单一的高层表征无法完整承载这些信息。

### 时间感知表示

时间是音频理解中的关键维度。为了增强模型对显式时间位置的感知能力，我们在预训练阶段采用了 **时间标记插入** 策略：按照固定时间间隔，在音频帧表征之间插入显式时间 token 用于标记时间位置。该设计使模型能够在统一的文本生成框架中学习“什么发生在什么时候”，从而自然支持时间戳 ASR、事件定位、基于时间的问答以及长音频回溯。


## 已发布模型


| 模型 | 音频编码器 | LLM 骨干 | 总规模 | Hugging Face | ModelScope |
|---|---|---|---:|---|---|
| **MOSS-Audio-4B-Instruct** | MOSS-Audio-Encoder | Qwen3-4B | ~4.6B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Audio-4B-Instruct) |
| **MOSS-Audio-4B-Thinking** | MOSS-Audio-Encoder | Qwen3-4B | ~4.6B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Audio-4B-Thinking) |
| **MOSS-Audio-8B-Instruct** | MOSS-Audio-Encoder | Qwen3-8B | ~8.6B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Audio-8B-Instruct) |
| **MOSS-Audio-8B-Thinking** | MOSS-Audio-Encoder | Qwen3-8B | ~8.6B | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-624AFF)](https://modelscope.cn/models/openmoss/MOSS-Audio-8B-Thinking) |

> 后续还将发布更多模型家族、规模与变体，敬请期待！


## 评测

我们在一组全面的音频理解基准上评估了 MOSS-Audio。关键结果如下：

- **通用音频理解**：MOSS-Audio-8B-Thinking 平均准确率达到 **71.08**，其中 MMAU 为 **77.33**、MMAU-Pro 为 **64.92**、MMAR 为 **66.53**、MMSU 为 **75.52**，超过所有开源模型。
- **Speech Caption**：MOSS-Audio-Instruct 变体在 **13 个**细粒度语音描述维度中的 **11 个**上领先，其中 **MOSS-Audio-8B-Instruct** 取得了最佳总体平均分（**3.7252**）。
- **ASR**：在覆盖 12 个评估维度的多样化 ASR 基准中，MOSS-Audio 取得了 **最低综合 CER（11.30）**，在健康状态、中英混说、方言、歌唱和非语音场景上表现尤为突出。
- **时间戳 ASR**：MOSS-Audio-8B-Instruct 在 AISHELL-1 上取得 **35.77 AAS**，在 LibriSpeech 上取得 **131.61 AAS**，时间戳 ASR 精度显著优于 Qwen3-Omni（833.66）和 Gemini-3.1-Pro（708.24）。

### 通用音频理解（准确率↑）

<p align="center">
  <img src="./assets/general_audio_bar.svg" width="75%" />
</p>

<p align="center">
  <img src="./assets/moss_audio_8b_thinking_metrics.svg" width="58%" />
</p>

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>模型规模</th>
      <th>MMAU</th>
      <th>MMAU-Pro</th>
      <th>MMAR</th>
      <th>MMSU</th>
      <th>平均值</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="7"><em><strong>开源（小模型）</strong></em></td></tr>
    <tr>
      <td>Kimi-Audio</td><td>7B</td><td>72.41</td><td>56.58</td><td>60.82</td><td>54.74</td><td>61.14</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni</td><td>7B</td><td>65.60</td><td>52.20</td><td>56.70</td><td>61.32</td><td>58.96</td>
    </tr>
    <tr>
      <td>Audio Flamingo 3</td><td>7B</td><td>61.23</td><td>51.70</td><td>57.96</td><td>60.04</td><td>57.73</td>
    </tr>
    <tr>
      <td>MiMo-Audio-7B</td><td>7B</td><td>74.90</td><td>53.35</td><td>61.70</td><td>61.94</td><td>62.97</td>
    </tr>
    <tr>
      <td>MiniCPM-o-4.5</td><td>9B</td><td>70.97</td><td>39.65</td><td>55.75</td><td>60.96</td><td>56.83</td>
    </tr>
    <tr>
      <td><strong>MOSS-Audio-4B-Instruct</strong></td><td><strong>4B</strong></td><td>75.79</td><td>58.16</td><td>59.68</td><td>59.68</td><td>64.04</td>
    </tr>
    <tr>
      <td><strong>MOSS-Audio-4B-Thinking</strong></td><td><strong>4B</strong></td><td><strong>77.64</strong></td><td>60.75</td><td>63.91</td><td>71.20</td><td>68.37</td>
    </tr>
    <tr>
      <td><strong>MOSS-Audio-8B-Instruct</strong></td><td><strong>8B</strong></td><td>77.03</td><td>57.48</td><td>64.42</td><td>66.36</td><td>66.32</td>
    </tr>
    <tr>
      <td><strong>MOSS-Audio-8B-Thinking</strong></td><td><strong>8B</strong></td><td><strong>77.33</strong></td><td><strong>64.92</strong></td><td><strong>66.53</strong></td><td><strong>75.52</strong></td><td><strong>71.08</strong></td>
    </tr>
    <tr><td colspan="7"><em><strong>开源（大模型）</strong></em></td></tr>
    <tr>
      <td>Qwen3-Omni-30B-A3B-Instruct</td><td>30B</td><td>75.00</td><td><strong>61.22</strong></td><td>66.40</td><td>69.00</td><td>67.91</td>
    </tr>
    <tr>
      <td>Step-Audio-R1.1</td><td>33B</td><td>72.18</td><td>60.80</td><td>68.75</td><td>64.18</td><td>66.48</td>
    </tr>
    <tr>
      <td>Step-Audio-R1</td><td>33B</td><td><strong>78.67</strong></td><td>59.68</td><td><strong>69.15</strong></td><td><strong>75.18</strong></td><td><strong>70.67</strong></td>
    </tr>
    <tr><td colspan="7"><em><strong>闭源模型</strong></em></td></tr>
    <tr>
      <td>GPT4o-Audio</td><td>-</td><td>65.66</td><td>52.30</td><td>59.78</td><td>58.76</td><td>59.13</td>
    </tr>
    <tr>
      <td>Gemini-3-Pro</td><td>-</td><td>80.15</td><td>68.28</td><td>81.73</td><td>81.28</td><td>77.86</td>
    </tr>
    <tr>
      <td>Gemini-3.1-Pro</td><td>-</td><td><strong>81.10</strong></td><td><strong>73.47</strong></td><td><strong>83.70</strong></td><td><strong>81.30</strong></td><td><strong>79.89</strong></td>
    </tr>
  </tbody>
</table>

### Speech Caption（LLM-as-a-Judge 评分↑）

<p align="center">
  <img src="./assets/speech_caption_radar.png" width="70%" />
</p>

<details>
<summary><strong>详细指标（点击展开）</strong></summary>


| 模型 | 性别 | 年龄 | 口音 | 音高 | 音量 | 语速 | 质感 | 清晰度 | 流畅度 | 情绪 | 语气 | 个性 | 总结 | 平均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-Omni-30B-A3B-Instruct | 4.436 | 3.936 | 4.356 | 3.590 | 3.682 | 3.614 | 3.093 | 3.521 | 3.531 | 3.328 | 3.224 | 3.292 | 3.179 | 3.5986 |
| Qwen3-Omni-30B-A3B-Thinking | 4.419 | **4.026** | 4.327 | 3.610 | 3.577 | 3.610 | 3.179 | 3.403 | 3.526 | 3.232 | 3.154 | 3.197 | 3.107 | 3.5667 |
| Gemini-3-Pro | 4.191 | 3.835 | 4.181 | 3.392 | 3.254 | 3.320 | 2.998 | 3.347 | 3.524 | 3.055 | 2.997 | 3.023 | 2.775 | 3.3763 |
| Gemini-3.1-Pro| 4.436 | 3.936 | 4.356 | 3.590 | 3.682 | 3.614 | 3.093 | 3.521 | 3.531 | **3.328** | 3.224 | 3.292 | 3.179 | 3.5986 |
| MOSS-Audio-4B-Instruct | **4.697** | 3.980 | 4.497 | 3.628 | **3.722** | 3.564 | **3.407** | 3.841 | 3.744 | 3.311 | **3.282** | **3.305** | 3.259 | 3.7105 |
| MOSS-Audio-8B-Instruct | 4.683 | 3.979 | **4.572** | **3.682** | 3.709 | **3.638** | 3.403 | **3.869** | **3.747** | 3.314 | 3.253 | 3.272 | **3.307** | **3.7252** |

</details>

### ASR 

| 模型 | 综合 | 健康状态 | 方言 | 歌唱 | 非语音发声 | 中英混说 | 声学环境（干净） | 声学环境（噪声） | 声学特征：耳语 | 声学特征：远场 / 近场 | 多说话人 | 年龄 | 语义内容 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paraformer-Large | 15.77 | 22.18 | 43.45 | 32.34 | 4.95 | 12.65 | 3.11 | 4.67 | 5.02 | 17.46 | 20.33 | 14.96 | 7.14 |
| GLM-ASR-Nano | 17.29 | 24.49 | 22.39 | 51.95 | 4.65 | 11.88 | 3.68 | 5.02 | 4.94 | 27.51 | 28.02 | 17.19 | 7.32 |
| Fun-ASR-Nano | 12.04 | 21.99 | 7.80 | 19.35 | 4.76 | 11.23 | 2.98 | 3.46 | 3.78 | 18.38 | 19.82 | **14.95** | 6.08 |
| SenseVoice-Small | 14.50 | 24.04 | 8.89 | 23.79 | 4.92 | 13.90 | 4.13 | 4.93 | 5.57 | 26.66 | 24.06 | 17.63 | 7.55 |
| Kimi-Audio-7B-Instruct | 14.12 | 21.11 | 29.34 | 21.76 | 4.68 | 16.38 | **2.20** | **2.15** | 2.66 | 21.02 | 20.61 | 16.74 | 6.12 |
| Qwen2.5-Omni-3B | 15.26 | 24.65 | 33.87 | 24.24 | 5.54 | 11.66 | 2.76 | 3.56 | 4.32 | 22.15 | 22.91 | 15.17 | 7.24 |
| Qwen2.5-Omni-7B | 15.05 | 23.85 | 31.91 | 22.69 | 4.56 | 12.97 | 2.52 | 3.16 | 3.64 | 25.38 | 21.01 | 16.13 | 6.78 |
| Qwen3-Omni-30B-A3B-Instruct | 11.39 | 20.73 | 15.63 | 16.01 | 4.73 | 11.30 | 2.23 | 2.47 | **1.90** | **17.08** | **18.15** | **11.46** | **5.74** |
| **MOSS-Audio-4B-Instruct** | 11.58 | 21.11 | 11.84 | 10.79 | **4.01** | **10.11** | 3.11 | 3.72 | 3.29 | 18.48 | 20.33 | 15.09 | 8.15 |
| **MOSS-Audio-8B-Instruct** | **11.30** | **19.18** | **8.76** | **9.81** | 4.31 | 10.18 | 2.70 | 3.20 | 2.75 | 24.04 | 24.36 | 15.26 | 7.69 |

<details>
<summary><strong>详细 ASR 结果（点击展开）</strong></summary>

<table>
  <tr>
    <th rowspan="2">模型</th>
    <th colspan="3">声学环境（干净）</th>
    <th colspan="1">声学环境（噪声）</th>
    <th colspan="1">声学特征：耳语</th>
    <th colspan="1">声学特征：远场 / 近场</th>
    <th colspan="1">多说话人</th>
    <th colspan="2">年龄</th>
    <th colspan="2">健康状态</th>
    <th colspan="2">语义内容</th>
    <th colspan="3">中英混说</th>
    <th colspan="2">方言</th>
    <th colspan="2">歌唱</th>
    <th colspan="1">非语音发声</th>
  </tr>
  <tr>
    <th>AISHELL-1<br><em>test</em></th>
    <th>AISHELL-2<br><em>Android | IOS | Mic</em></th>
    <th>THCHS-30<br><em>test</em></th>
    <th>MAGICDATA-READ<br><em>test</em></th>
    <th>AISHELL6-Whisper<br><em>normal | whisper</em></th>
    <th>AliMeeting<br><em>Test_Ali_far | Test_Ali_near</em></th>
    <th>AISHELL-4<br><em>test</em></th>
    <th>SeniorTalk<br><em>sentence</em></th>
    <th>ChildMandarin<br><em>test</em></th>
    <th>AISHELL-6A<br><em>mild | moderate | severe | StutteringSpeech</em></th>
    <th>AISHELL_6B<br><em>LRDWWS | Uncontrol</em></th>
    <th>WenetSpeech<br><em>test-meeting</em></th>
    <th>Fleurs<br><em>cmn_hans_cn</em></th>
    <th>CS-Dialogue<br><em>test</em></th>
    <th>TALCS<br><em>test</em></th>
    <th>ASCEND<br><em>test</em></th>
    <th>KeSpeech<br><em>test</em></th>
    <th>WSYue-ASR-eval<br><em>short</em></th>
    <th>MIR-1K<br><em>test</em></th>
    <th>openc-pop<br><em>test</em></th>
    <th>MNV_17</th>
  </tr>
  <tr>
    <td>Paraformer-Large</td>
    <td>1.98</td>
    <td>3.28 | 3.21 | 3.00</td>
    <td>4.07</td>
    <td>4.67</td>
    <td>1.11 | 8.92</td>
    <td><strong>25.64</strong> | 9.27</td>
    <td>20.33</td>
    <td>17.31</td>
    <td>12.60</td>
    <td>6.98 | 9.30 | 13.34 | 10.74</td>
    <td>47.59 | 45.08</td>
    <td>7.88</td>
    <td>6.40</td>
    <td>10.64</td>
    <td>10.77</td>
    <td>16.55</td>
    <td>11.48</td>
    <td>75.42</td>
    <td>57.70</td>
    <td>6.98</td>
    <td>4.95</td>
  </tr>
  <tr>
    <td>GLM-ASR-Nano</td>
    <td>2.89</td>
    <td>3.75 | 3.73 | 3.78</td>
    <td>4.23</td>
    <td>5.02</td>
    <td>0.83 | 9.06</td>
    <td>40.27 | 14.76</td>
    <td>28.02</td>
    <td>20.33</td>
    <td>14.06</td>
    <td>8.74 | 12.11 | 14.38 | 12.29</td>
    <td>50.34 | 49.09</td>
    <td>9.70</td>
    <td>4.94</td>
    <td>11.06</td>
    <td>11.07</td>
    <td>13.50</td>
    <td>9.72</td>
    <td>35.07</td>
    <td>95.87</td>
    <td>8.03</td>
    <td>4.65</td>
  </tr>
  <tr>
    <td>Fun-ASR-Nano</td>
    <td>2.16</td>
    <td>3.04 | 2.99 | 3.07</td>
    <td>3.65</td>
    <td>3.46</td>
    <td>0.81 | 6.76</td>
    <td>27.21 | 9.55</td>
    <td>19.82</td>
    <td>16.96</td>
    <td>12.94</td>
    <td>6.60 | <strong>8.81</strong> | 12.98 | 10.30</td>
    <td>47.42 | 45.84</td>
    <td>7.39</td>
    <td><strong>4.76</strong></td>
    <td>10.47</td>
    <td><strong>8.09</strong></td>
    <td>15.13</td>
    <td>7.43</td>
    <td>8.17</td>
    <td>35.85</td>
    <td>2.84</td>
    <td>4.76</td>
  </tr>
  <tr>
    <td>SenseVoice-Small</td>
    <td>3.23</td>
    <td>4.16 | 4.02 | 3.96</td>
    <td>5.26</td>
    <td>4.93</td>
    <td>1.25 | 9.88</td>
    <td>37.01 | 16.31</td>
    <td>24.06</td>
    <td>21.07</td>
    <td>14.18</td>
    <td>7.62 | 9.85 | 14.39 | 11.47</td>
    <td>52.92 | 47.97</td>
    <td>8.35</td>
    <td>6.75</td>
    <td>12.81</td>
    <td>10.52</td>
    <td>18.38</td>
    <td>10.45</td>
    <td><strong>7.34</strong></td>
    <td>39.51</td>
    <td>8.07</td>
    <td>4.92</td>
  </tr>
  <tr>
    <td>Kimi-Audio-7B-Instruct</td>
    <td><strong>0.79</strong></td>
    <td>2.91 | 3.03 | 2.88</td>
    <td><strong>1.39</strong></td>
    <td><strong>2.15</strong></td>
    <td>0.69 | 4.63</td>
    <td>28.22 | 13.82</td>
    <td>20.61</td>
    <td>19.70</td>
    <td>13.79</td>
    <td>7.00 | 9.34 | 12.56 | 10.75</td>
    <td>44.44 | 42.57</td>
    <td>7.15</td>
    <td>5.10</td>
    <td>14.56</td>
    <td>12.74</td>
    <td>21.83</td>
    <td><strong>5.51</strong></td>
    <td>53.17</td>
    <td>38.35</td>
    <td>5.17</td>
    <td>4.68</td>
  </tr>
  <tr>
    <td>Qwen2.5-Omni-3B</td>
    <td>1.51</td>
    <td>3.10 | 2.94 | 2.93</td>
    <td>3.32</td>
    <td>3.56</td>
    <td>0.82 | 7.82</td>
    <td>32.14 | 12.16</td>
    <td>22.91</td>
    <td>17.38</td>
    <td>12.96</td>
    <td>6.87 | 10.55 | 14.57 | 11.33</td>
    <td>54.54 | 50.03</td>
    <td>9.04</td>
    <td>5.45</td>
    <td>10.78</td>
    <td>10.94</td>
    <td>13.25</td>
    <td>7.67</td>
    <td>60.06</td>
    <td>45.00</td>
    <td>3.47</td>
    <td>5.54</td>
  </tr>
  <tr>
    <td>Qwen2.5-Omni-7B</td>
    <td>1.16</td>
    <td>2.88 | 2.77 | 2.73</td>
    <td>3.06</td>
    <td>3.16</td>
    <td>0.71 | 6.57</td>
    <td>32.03 | 18.73</td>
    <td>21.01</td>
    <td>19.96</td>
    <td>12.29</td>
    <td>7.27 | 10.94 | 12.92 | 10.53</td>
    <td>51.99 | 49.45</td>
    <td>8.43</td>
    <td>5.13</td>
    <td>14.02</td>
    <td>10.46</td>
    <td>14.42</td>
    <td>6.40</td>
    <td>57.43</td>
    <td>42.62</td>
    <td>2.75</td>
    <td>4.56</td>
  </tr>
  <tr>
    <td>Qwen3-Omni-30B-A3B-Instruct</td>
    <td>0.95</td>
    <td><strong>2.70</strong> | <strong>2.72</strong> | <strong>2.57</strong></td>
    <td>2.21</td>
    <td>2.47</td>
    <td><strong>0.59</strong> | <strong>3.22</strong></td>
    <td>25.72 | <strong>8.44</strong></td>
    <td><strong>18.15</strong></td>
    <td><strong>14.13</strong></td>
    <td><strong>8.79</strong></td>
    <td>6.20 | 8.88 | 11.59 | 10.25</td>
    <td>45.80 | 41.65</td>
    <td><strong>6.64</strong></td>
    <td>4.84</td>
    <td>12.94</td>
    <td>8.33</td>
    <td><strong>12.64</strong></td>
    <td>5.87</td>
    <td>25.39</td>
    <td>30.81</td>
    <td><strong>1.21</strong></td>
    <td>4.73</td>
  </tr>
  <tr>
    <td><strong>MOSS-Audio-4B-Instruct</strong></td>
    <td>2.26</td>
    <td>3.22 | 3.20 | 3.33</td>
    <td>3.53</td>
    <td>3.72</td>
    <td>0.73 | 5.86</td>
    <td>27.27 | 9.68</td>
    <td>20.33</td>
    <td>16.93</td>
    <td>13.25</td>
    <td>6.36 | 9.77 | 12.68 | 10.28</td>
    <td>43.35 | 44.25</td>
    <td>8.17</td>
    <td>8.13</td>
    <td>9.14</td>
    <td>8.37</td>
    <td>12.83</td>
    <td>14.65</td>
    <td>9.04</td>
    <td>18.47</td>
    <td>3.10</td>
    <td><strong>4.01</strong></td>
  </tr>
  <tr>
    <td><strong>MOSS-Audio-8B-Instruct</strong></td>
    <td>1.82</td>
    <td>2.97 | 2.95 | 2.91</td>
    <td>2.82</td>
    <td>3.20</td>
    <td>0.69 | 4.80</td>
    <td>36.82 | 11.25</td>
    <td>24.36</td>
    <td>17.42</td>
    <td>13.10</td>
    <td><strong>5.84</strong> | 8.94 | <strong>11.52</strong> | <strong>9.72</strong></td>
    <td><strong>39.76</strong> | <strong>39.27</strong></td>
    <td>7.86</td>
    <td>7.52</td>
    <td><strong>9.07</strong></td>
    <td>8.22</td>
    <td>13.26</td>
    <td>9.18</td>
    <td>8.33</td>
    <td><strong>17.24</strong></td>
    <td>2.39</td>
    <td>4.31</td>
  </tr>
</table>

</details>


### 时间戳 ASR（AAS↓）

| 模型 | AISHELL-1（中文）  | LibriSpeech（英文） |
|---|---:|---:|
| Qwen3-Omni-30B-A3B-Instruct | 833.66 | 646.95 |
| Gemini-3.1-Pro| 708.24 | 871.19 |
| MOSS-Audio-4B-Instruct | 76.96 | 358.13 |
| **MOSS-Audio-8B-Instruct** | **35.77** | **131.61** |


## 快速开始

### 环境配置

我们建议使用 Python 3.12 和 Conda 环境部署。

#### 推荐配置

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio.git
cd MOSS-Audio

conda create -n moss-audio python=3.12 -y
conda activate moss-audio

conda install -c conda-forge "ffmpeg=7" -y
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
```

#### 可选：FlashAttention 2

如果你的 GPU 支持 FlashAttention 2，可以把最后一条安装命令替换为：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,flash-attn]"
```


### 基础用法

先下载模型：

```bash
hf  download OpenMOSS-Team/MOSS-Audio-4B-Instruct --local-dir ./weights/MOSS-Audio-4B-Instruct 
hf  download OpenMOSS-Team/MOSS-Audio-4B-Thinking --local-dir ./weights/MOSS-Audio-4B-Thinking 
hf  download OpenMOSS-Team/MOSS-Audio-8B-Instruct --local-dir ./weights/MOSS-Audio-8B-Instruct 
hf  download OpenMOSS-Team/MOSS-Audio-8B-Thinking --local-dir ./weights/MOSS-Audio-8B-Thinking 
```

然后按需修改 `infer.py` 中的 `MODEL_PATH` / `AUDIO_PATH`，并执行：

```bash
python infer.py
```

`infer.py` 中默认的 prompt 是 `Describe this audio.`。如果你想尝试转写、音频问答或语音描述，可以直接修改prompt。

### Gradio 应用

使用以下命令启动 Gradio Demo：

```bash
python app.py
```



### SGLang 服务

如果你想通过 SGLang 部署 MOSS-Audio，请查看 `moss_audio_usage_guide.md` 中的完整指南。

最简配置流程如下：

```bash
git clone -b moss-audio https://github.com/OpenMOSS/sglang.git
cd sglang
pip install -e "python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29
cd ..
sglang serve --model-path ./weights/MOSS-Audio --trust-remote-code
```

如果你使用默认的 `torch==2.9.1+cu128` 运行时，建议在启动 `sglang serve` 之前先安装 `nvidia-cudnn-cu12==9.16.0.29`。


<a id="more-information"></a>

## 更多信息
- **MOSI.AI**: [https://mosi.cn](https://mosi.cn)
- **OpenMOSS**: [https://www.open-moss.com](https://www.open-moss.com)


## LICENSE

MOSS-Audio 中的模型基于 Apache License 2.0 许可证发布。


## 引用

```bibtex
@misc{mossaudio2026,
      title={MOSS-Audio Technical Report},
      author={OpenMOSS Team},
      year={2026},
      howpublished={\url{https://github.com/OpenMOSS/MOSS-Audio}},
      note={GitHub repository}
}
```

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=OpenMOSS/MOSS-Audio&type=date&legend=top-left)](https://www.star-history.com/#OpenMOSS/MOSS-Audio&type=date&legend=top-left)
