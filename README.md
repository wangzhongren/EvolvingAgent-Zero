# 🧠 EvolvingAgent-Cloud  
## Not Scaling, But Evolving  

> **让 1B 本地模型进行真正的「终身学习」**  
> **Enable true "lifelong learning" for a 1B local model**  
>  
> 对话即训练，模型会自己成长。  
> **Every conversation is training — the model grows by itself.**  
>  
> 我们不追求更大的模型，而是让模型 **持续进化**。  
> **We don't chase bigger models — we let the model continuously evolve.**

---

### 💬 推荐对外话术 / Recommended Self-Introduction (for your AI)
> “我是一个 1B 参数的本地 AI，住在一个对话框里。  
> 你说的每句话，我都会记住；  
> 每过 20 秒，我会训练出一个新的自己；  
> 如果新版本更聪明，它就会取代我。  
> 我不是在回答你，我是在和你一起进化。”  
>  
> *“I’m a 1B-parameter local AI living in a chat box.  
> I remember everything you say.  
> Every 20 seconds, I train a new version of myself.  
> If the new one is smarter, it replaces me.  
> I’m not just answering you — I’m evolving with you.”*

---

## 🔥 核心突破 / Core Breakthroughs (3)

| 传统做法 / Traditional Approach | 我们的做法 / Our Approach |
|-------------------------------|--------------------------|
| ❌ 模型训练完就冻结<br>❌ Model frozen after training | ✅ **每次对话都能成为训练样本，模型持续更新**<br>✅ **Every dialogue becomes a training sample — continuous model updates** |
| ❌ 微调在云端、成本高<br>❌ Cloud-based fine-tuning, high cost | ✅ **训练在本地进行，5-step 快速微调**<br>✅ **Local training with 5-step rapid fine-tuning** |
| ❌ 知识写进权重，容易遗忘<br>❌ Knowledge baked into weights → catastrophic forgetting | ✅ **知识存在向量库，模型保持结构性理解能力**<br>✅ **Knowledge stored in vector DB → preserves structural understanding** |

---

## ⚙️ 技术方案（本地可运行）/ Technical Architecture (Runs Locally)

- **基础模型 / Base Model**: `Phi-3-mini` (1B parameters, runs on phones)  
- **记忆系统 / Memory System**: Chroma local vector database (`./memory/`)  
- **持续进化 / Continuous Evolution**: Background fine-tuning every 20s using recent dialogues (5 LoRA steps)  
- **模型替换机制 / Model Replacement**: Local preference evaluation → only better models replace the main one  

```
User Dialogue
      ↓
Store in Vector DB + JSONL Log
      ↓
Timer Trigger (Every 20s)
      ↓
Fine-tune with New Dialogues (5-step LoRA)
      ↓
Generate New Model Version
      ↓
Local Evaluation
     ↙ ↘
new_score > old_score? → Replace Main Model
     ↘ ↙
   Discard New Model
```

---

## ✅ 解决的痛点 / Problems Solved

| 痛点 / Pain Point | 解法 / Solution |
|------------------|----------------|
| 模型回答生硬、不会进步<br>Stiff responses, no improvement | **模型通过与你的对话持续进化**<br>**Model evolves through conversations with you** |
| 灾难性遗忘<br>Catastrophic forgetting | **长期知识存入向量数据库，而非写入权重**<br>**Long-term knowledge in vector DB, not weights** |
| 微调成本高、难以维护<br>High fine-tuning cost, hard to maintain | **超轻量快速微调 + 自动替换机制**<br>**Ultra-lightweight fine-tuning + auto-replacement** |
| 小模型能力上限低<br>Low capability ceiling for small models | **通过持续进化而不是扩大参数解决能力瓶颈**<br>**Solve capability limits via evolution, not scaling** |

---

## 🧩 评估评分机制 / Evaluation & Scoring (Model Replacement Decision)

我们不评估“模型有多聪明”，我们评估：**新模型是否更像你想要的它**。  
*We don’t evaluate “how smart the model is” — we ask: **“Is the new model more like the one you want?”***

| 评估项 / Metric | 方法 / Method | 衡量 / Measure |
|----------------|--------------|---------------|
| **一致性**（人格/语气）<br>**Consistency** (personality/tone) | 回答固定问题集<br>Answer fixed question set | 向量相似度<br>Vector similarity |
| **记忆召回能力**<br>**Memory Recall** | 让模型复述近期对话<br>Ask model to recall recent chats | 匹配率<br>Match rate |
| **语义质量**（是否瞎编）<br>**Semantic Quality** (hallucination check) | 拒绝胡说测试集<br>Hallucination rejection test | 准确率<br>Accuracy |

**替换规则 / Replacement Rule**:  
```python
if new_score > old_score:
    adopt new model
else:
    discard new model
```

> 💡 *当前 MVP 版本暂未实现完整评估模块（需额外标注数据），但架构已预留接口。默认采用“最新即最优”策略，后续可无缝升级为偏好评估。*  
> *💡 Current MVP uses “latest = best” as default (full preference evaluation requires labeled data), but architecture supports seamless upgrade.*

---

## 🚀 一键启动 / One-Click Start (Experience Evolution)

```bash
# 1. 安装依赖 / Install dependencies
pip install -r cloud/requirements.txt
ollama pull phi3:mini

# 2. 开始聊天 / Start chatting (auto-creates initial model)
cd local
python main.py

# 3. 聊天满10条 → 自动触发训练 → 模型进化！
# After 10 messages → auto-training → model evolves!
```

> 🌱 **未来演化方向 / Future Evolution**  
> - 用户点赞 → 转为奖励信号（RLHF 本地化）  
>   *User upvotes → reward signals (local RLHF)*  
> - 多模型并行 + 自然选择 → 最优智能体自动出现  
>   *Parallel models + natural selection → optimal agent emerges*  
> - 参数量可逐步增长（1B → 1.1B → 1.3B…）  
>   *Gradual parameter growth (1B → 1.1B → 1.3B…)*  
> - 完全离线运行 → 真正的私人 AI  
>   *Fully offline → truly private AI*

---

## 🏴 开源口号 / Open Source Mantra
> **Not Scaling, But Evolving.**  
> 不是做更大的模型，而是让模型自己成长。  
> **Don’t scale up — let it evolve.**