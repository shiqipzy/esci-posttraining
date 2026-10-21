# modeling.py：模型加载、四分类头、推理与存档（96 行）

> 一句话：这是全项目唯一直接接触 Qwen 权重与 LoRA 的文件。它把“一条文本”变成“四个分数”，再把四个分数变成“四类概率 + 一个排序分”，并把训练好的 adapter 存成可复用的目录。

下文模型构造示例以第二阶段续训为主：它加载已有 S adapter；第一阶段 SFT 则执行新建 `LoraConfig` 的分支。两阶段共用相同的前向、保存和预测函数。

**读完这一篇，你应该能回答：**

1. 为什么 `forward()` 只算四列，而不是让模型输出整个词表？
2. “取最后一个有效位置”具体是哪一个 token？为什么它对应类别字母？
3. 为什么本轮永远不会执行 `get_peft_model(LoraConfig(...))` 那段？
4. `infer()` 和 `predict()` 的分工是什么？`score` 怎么算？

---

## 1. 角色与导入策略

```python
"""Qwen four-label classification, LoRA adapters and training losses."""
from pathlib import Path
import torch
import torch.nn.functional as F
from common import GAINS, LABELS, collate, digest, encode, verify_model, write_json
```

顶层只 import `torch`；`transformers` 和 `peft` 是在 `RelevanceModel.__init__()` 内部才 import 的（第 11、22 行）。这不是随手写的：

- `fusion.py` **不在顶层** import `modeling`，而是在 `load_model()`、`infer()` 等函数内部 import。
- 所以 `python src/train.py fusion --help` 只加载 `common` / `fusion` / `evaluation`，**不需要装 PyTorch**，也不需要 1.5 GB 的底座。
- 真正要加载模型时，才会去 import `transformers` 和 `peft`。

这种“把重依赖推迟到真正使用的那一刻”的写法，让预检、审计、命令行帮助在没有 GPU 的机器上也能跑。

模型前向只产生 `[B,4]` logits；SFT 的 CE 在 `sft.train_sft()` 中，续训损失在 `fusion.loss_rows()` 中。

---

## 2. 构造 `__init__()`（第 9–43 行）

```python
def __init__(self, source, config, device='cuda', trainable=False, checkpoint=None):
    super().__init__()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    self.run_config = config
    self.provenance = verify_model(str(Path(source).resolve()))
    self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    self.tokenizer.padding_side = 'right'
    ...
```

参数含义：

| 参数 | 本轮取值 | 作用 |
| --- | --- | --- |
| `source` | `ROOT / cfg['base']`，即 `models/base` | 底座权重目录（官方 Qwen3-0.6B） |
| `config` | 整份 `configs/fusion_continue_20261021.json` | 训练时会读 `lora_r` 等键；本轮重建 LoRA 的分支不执行 |
| `device` | 默认 `'cuda'` | 决定精度与张量位置 |
| `trainable` | 训练时 `True`，评测时 `False` | 是否允许梯度、是否 `eval()` |
| `checkpoint` | `models/S` 或 `out/<arm>/step-XXXX` | 指定就加载该 adapter；不指定且可训练才新建 LoRA |

### 2.1 先核对身份，再加载（第 12–14 行）

`verify_model(str(Path(source).resolve()))` 在真正 `from_pretrained` 之前先跑一遍（见 [01 §4.1](01-common.md)）：文件大小 + Git blob / LFS 哈希。**身份不对就不加载**，避免用一份来路不明的底座训练出一份无法复现的结果。

`self.run_config = config` 只是把配置挂在对象上，`save()` 时会原样写进 `run_config.json`，方便事后知道这个 adapter 是在什么参数下训出来的。

分词器用 `local_files_only=True`、`trust_remote_code=False`：只读本地文件、不执行仓库里的自定义代码。`padding_side='right'` 与 `collate()` 在右边补齐一致，保证“最后一个有效位置”是真正的末尾。

### 2.2 补齐 pad token（第 16–17 行）

```python
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
```

Qwen 的分词器默认没有单独的 `pad_token`。`collate()` 需要一个 id 来填充短序列；这里退而使用 `eos` 的 id。因为 `attention_mask` 把这些位置标成 0，模型不会把它们当成内容，只借用这个 id 让张量成矩形。

### 2.3 精度：BF16 是硬要求（第 18–21 行）

```python
        self.amp_enabled = str(device).startswith('cuda')
        if self.amp_enabled and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('This frozen GPU recipe requires CUDA BF16. Do not silently switch precision.')
        dtype = torch.bfloat16 if self.amp_enabled else torch.float32
```

- `device='cuda'` → 权重用 `bfloat16`；`device='cpu'` → `float32`。
- 在 CUDA 上如果显卡不支持 BF16，**直接报错**，不悄悄退回 FP16/FP32。报错信息直说 `Do not silently switch precision`：换精度会改变数值结果，已发布指标就不再可比。
- 实际运行环境是 RTX 4080 SUPER，支持 BF16。

注意这里只判断“设备名以 cuda 开头”，所以 `device='cuda:0'` 也算。真正创建张量时 PyTorch 默认用 0 号卡。

### 2.4 加载底座与四个标签 token（第 22–29 行）

```python
        from peft import LoraConfig, PeftModel, get_peft_model
        self.backbone = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=dtype, attn_implementation='sdpa')
        self.backbone.config.use_cache = False
        ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
        if not all(len(x) == 1 for x in ids):
            raise ValueError('Labels must be single tokens')
        self.register_buffer('label_ids', torch.tensor([x[0] for x in ids]), persistent=False)
```

- **`dtype=dtype`**：直接以 bf16 加载，避免“先 fp32 加载再转”的显存峰值。新版 transformers 用 `dtype` 参数（旧版叫 `torch_dtype`）。
- **`attn_implementation='sdpa'`**：使用 PyTorch 的 fused scaled-dot-product attention，比 eager 快、比 flash-attention 依赖少。
- **`use_cache = False`**：分类任务不需要自回归生成，关掉 KV cache 省显存。
- **四个标签必须各是一个 token**：`tokenizer.encode(label, add_special_tokens=False)` 对 `E`、`S`、`C`、`I` 分别编码，长度必须都是 1。这是整个“四分类头”的前提：如果某个字母被切成两个 token，就没法用“一个位置上的四个候选 token”来表达四类。
- **`register_buffer('label_ids', ..., persistent=False)`**：把四个 token id 注册成张量缓冲区，`.to(device)` 时会跟着移动到 GPU；`persistent=False` 表示**不写进 `state_dict`**。这是有意的：adapter 的权重文件不该包含这个派生出来的索引；加载时按当前分词器重新算即可。

`label_ids` 的形状是 `[4]`，内容大致是 Qwen 词表里大写字母 E/S/C/I 各自对应的 id。

### 2.5 两个分支：加载已有 adapter，或新建 LoRA（第 30–36 行）

```python
        if checkpoint:
            self.backbone = PeftModel.from_pretrained(self.backbone, str(Path(checkpoint) / 'adapter'),
                                                      is_trainable=trainable)
        elif trainable:
            self.backbone = get_peft_model(self.backbone, LoraConfig(task_type='CAUSAL_LM',
                r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                target_modules=config['target_modules'], bias='none'))
```

**本轮只走第一条分支。** 三组的 `load_model()` 都传了 `checkpoint`：

- 起点：`models/S`（内含 `adapter/`）；
- 恢复：`out/<arm>/step-XXXX`（内含 `adapter/`）。

因此 `elif trainable` 那段**一次也不会执行**。这解释了两个看起来奇怪的地方：

1. 配置 [fusion_continue_20261021.json](../../configs/fusion_continue_20261021.json) 里**没有** `lora_r`、`lora_alpha`、`target_modules` 这些键，代码却引用了它们——因为引用发生在不会执行的路径上。
2. LoRA 的超参数实际来自 models/S/adapter/adapter_config.json（本地文件 `models/S/adapter/adapter_config.json`，未随仓库发布）：`r=16`、`lora_alpha=32`、`lora_dropout=0.05`、`target_modules=['k_proj','q_proj','up_proj','v_proj','o_proj','gate_proj','down_proj']`。加载时以文件里的值为准。

`PeftModel.from_pretrained(..., is_trainable=trainable)` 的两个作用：把 adapter 权重注入底座；当 `is_trainable=True` 时把 LoRA 参数设为可训练。若从零新建 LoRA，`get_peft_model()` 会按 `LoraConfig` 插入低秩矩阵。

> 那份 `adapter_config.json` 里的 `base_model_name_or_path` 是 `/root/base_sft/model`——训练 S 时的远端路径。本地加载只用它做记录，实际底座由 `source` 参数决定，所以路径不存在也不影响。

### 2.6 可训练性与梯度检查点（第 37–43 行）

```python
        if trainable and config['gradient_checkpointing']:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.backbone.enable_input_require_grads()
        self.to(device)
        if not trainable:
            self.requires_grad_(False)
            self.eval()
```

- 本轮配置 `gradient_checkpointing=false`，所以前两行不执行；打开它会用“重算前向”换显存，`use_reentrant=False` 是新版推荐写法。
- `self.to(device)` 把整个模型（含 `label_ids` 缓冲区）搬到目标设备。
- 评测时 `requires_grad_(False)` 关闭所有梯度，并切到 `eval()` 模式。
- 训练时**不**显式调用 `model.train()`，而是由 `train_arm()` 每步调用（见 [04 §8](04-training.md)）。

---

## 3. 四分类头：`forward()`（第 50–58 行）

整个文件最核心的 8 行：

```python
    def forward(self, features):
        # LoRA modules are installed inside the underlying Qwen layers. Avoid materializing
        # [batch, sequence, vocabulary] logits; project only the last valid state to four labels.
        base = self.base_model()
        out = base.model(**features, use_cache=False).last_hidden_state
        last = features['attention_mask'].sum(dim=-1) - 1
        h = out[torch.arange(out.shape[0], device=out.device), last].float()
        weights = base.lm_head.weight.index_select(0, self.label_ids).float()
        return F.linear(h, weights)
```

### 3.1 逐行与形状

设批大小 `B`、序列长度 `L`、隐藏维度 `H`（Qwen3-0.6B 是 1024）：

| 行 | 表达式 | 形状 / 类型 | 说明 |
| --- | --- | --- | --- |
| 54 | `base.model(**features, use_cache=False).last_hidden_state` | `[B, L, H]`，bf16 | 只跑 Transformer 主体，**不跑 lm_head** |
| 55 | `features['attention_mask'].sum(dim=-1) - 1` | `[B]`，long | 每条序列最后一个有效位置的下标 |
| 56 | `out[arange(B), last]` → `.float()` | `[B, H]`，fp32 | 按行取“最后一个有效位置”的隐藏表示 |
| 57 | `base.lm_head.weight.index_select(0, self.label_ids)` | `[4, H]`，fp32 | 词表输出层里 E/S/C/I 四行权重 |
| 58 | `F.linear(h, weights)` | `[B, 4]`，fp32 | 每个样本四个原始分数（logits） |

`F.linear(h, W)` 就是 `h @ W.T`，得到“每个样本与四个标签 token 的匹配分数”。

Qwen3-0.6B 的 `config.json` 里 `tie_word_embeddings` 为 `true`，`lm_head.weight` 与输入嵌入共享同一份存储。所以“取四行输出权重”同时也等价于“取 E、S、C、I 四个 token 的嵌入向量”。无论共享与否，这四行都**冻结**：LoRA 的 `target_modules` 只包含注意力与 MLP 的投影层，不含 `lm_head`。

### 3.2 为什么取 “mask 求和 - 1”

`collate()` 在右边补齐，`attention_mask` 前段全 1、后段全 0。第 i 行 1 的个数就是真实 token 数，减 1 得到**最后一个真实 token 的下标**。

这个 token 是什么？`encode()` 用 `add_generation_prompt=True` 渲染对话，结尾是 `<|im_start|>assistant\n`。所以最后一个位置正好是“助手该开口”的位置——它之后的下一个 token 就是答案字母。`forward()` 拿这个位置的隐藏表示去和四个标签 token 的权重做点积，等价于问模型：“下一个 token 最可能是 E、S、C、I 中的哪一个？”

### 3.3 为什么不直接输出整个词表

如果写成常见的 `base(**features).logits`，会得到 `[B, L, V]`，其中 `V ≈ 151,936`（Qwen 词表大小）。即使只取最后一个位置 `logits[:, -1, :]`，也要先把整个 `[B, L, V]` 物化出来——以 `B=4, L=180` 估算就是 4×180×151936 ≈ 1.09 亿个数，bf16 下约 218 MB，而其中 99.997% 的列会被丢弃。

这里改成“先取 `[B, H]` 的隐藏表示，再乘 `[4, H]` 的权重”，只做一次小矩阵乘法。注释里的 “Avoid materializing [batch, sequence, vocabulary] logits” 说的就是这件事。训练时这还带来一个额外好处：反向传播只需要为这 4 列计算，而不是为 15 万列。

### 3.4 为什么最后 `.float()`

底座在 bf16 下计算，但四个分数的比较、softmax、KL 对数值精度敏感。把隐藏表示和权重都转成 fp32 再做点积，logits 就是 fp32；后续 `loss_rows()` 里又对 logits 做一次 `.float()`（幂等）。训练记录里 `loss_dtype: float32` 指的就是这一层。

---

## 4. `base_model()`：剥掉 LoRA 包装（第 45–48 行）

```python
    def base_model(self):
        # PeftModel wraps the Qwen model; a bare backbone is already the base model.
        getter = getattr(self.backbone, 'get_base_model', None)
        return getter() if getter is not None else self.backbone
```

`self.backbone` 可能是两种东西：

- 加载了 adapter 时：`PeftModel` 包装对象，真正的 Qwen 在它内部，用 `get_base_model()` 取出；
- 没有 adapter 时：裸的 `Qwen3ForCausalLM`，本身就是 base。

`getattr(..., None)` + 分支的写法避免了对 `peft` 的硬依赖（如果环境里没装 peft，裸模型也能用）。`forward()` 需要 base，是因为要在**底座**上调用 `.model`（Transformer 主体）和取 `.lm_head`（输出层），而 `PeftModel` 转发这些属性不一定完整。

---

## 5. `save()`：一个存档目录里有什么（第 60–68 行）

```python
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(directory / 'adapter', safe_serialization=True)
        self.tokenizer.save_pretrained(directory / 'tokenizer')
        write_json(directory / 'run_config.json', self.run_config)
        write_json(directory / 'checkpoint_manifest.json', {'files': {
            str(p.relative_to(directory)).replace('\\', '/'): digest(p)
            for p in directory.rglob('*') if p.is_file()}})
```

四样东西：

1. **`adapter/`**：LoRA 权重（`adapter_model.safetensors`）+ `adapter_config.json`。用 `safe_serialization=True` 存成 safetensors，不执行 pickle。
2. **`tokenizer/`**：分词器文件。存档自带分词器，推理时不必再回到底座目录取。
3. **`run_config.json`**：构造时传入的整份配置。
4. **`checkpoint_manifest.json`**：目录内每个文件的相对路径 → SHA256。路径统一用 `/` 分隔，避免 Windows 写出的 `\` 让哈希清单在不同平台上不一致。

**一个容易误会的点**：`rglob('*')` 只扫“此刻已经存在”的文件。`training_state.pt` 和 `state.json` 是稍后由 `snapshot()` 写进去的（见 [04 §7](04-training.md)），所以它们**不在** `checkpoint_manifest.json` 里。这个清单描述的是“模型文件”，不是整个目录的完整清单。

`mkdir(..., exist_ok=False)` 也值得注意：它拒绝写进已存在的目录。`snapshot()` 会先写一个带随机后缀的 `.pending-*` 目录，再整体重命名，正是为了配合这个语义。

---

## 6. 推理：`infer()` 与 `predict()`（第 71–96 行）

### 6.1 `infer()`：只负责概率（第 71–83 行）

```python
def infer(model, rows, batch_size, max_length=512, demonstrations=()):
    """Return (four probabilities, input token count) per row, in row order."""
    features = [encode(r, model.tokenizer, max_length, demonstrations) for r in rows]
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = collate(features[start:start + batch_size], model.tokenizer.pad_token_id, device)
            probabilities = model(batch).softmax(-1).cpu().tolist()
            for offset, ps in enumerate(probabilities):
                result.append((ps, len(features[start + offset]['input_ids'])))
    return result
```

- 先把**所有**行编码好（`features`），再按 `batch_size` 切块前向。编码是纯 CPU 的 Python 循环，先做完可以让 GPU 阶段的数据搬运更规律。
- `device = next(model.parameters()).device`：不写死 `'cuda'`，模型在哪就在哪推理。
- `model.eval()` + `torch.inference_mode()`：关闭 dropout、关闭梯度记录。`inference_mode` 比 `no_grad` 更彻底（不记录版本计数），适合纯推理。
- `model(batch)` 得到 `[B, 4]` logits，`.softmax(-1)` 沿类别维归一化，`.cpu().tolist()` 变回 Python 浮点数列表。
- 返回值是**按行顺序**排列的 `(四类概率列表, 输入 token 数)`。token 数是整条序列长度（系统提示 + 用户消息 + 生成提示），不是商品文本本身的长度。F 上实测：最小 131、中位数 165、最大 245。

`softmax(-1)` 是唯一把 logits 变成概率的地方。**训练时的 `loss_rows()` 里没有 softmax**，而是直接对 logits 做 `log_softmax`；两者在数学上一致（`log_softmax` 后再 `exp` 就是 softmax），这样损失可以数值稳定地算。

### 6.2 `predict()`：把概率整理成可写文件的行（第 86–96 行）

```python
def predict(model, rows, batch_size, max_length=512, demonstrations=()):
    result = []
    for row, (probabilities, tokens) in zip(rows, infer(model, rows, batch_size, max_length, demonstrations)):
        result.append({**{k: row[k] for k in ('example_id', 'query_id', 'product_id', 'query', 'title', 'brand',
                                              'title_clipped', 'brand_clipped')},
                       'gold': row['label'],
                       'score': sum(p * g for p, g in zip(probabilities, GAINS)),
                       'prediction': LABELS[max(range(4), key=lambda i: probabilities[i])],
                       'input_tokens': tokens,
                       'probabilities': dict(zip(LABELS, probabilities))})
    return result
```

每一行输出包含：

| 字段 | 来源 | 含义 |
| --- | --- | --- |
| `example_id` / `query_id` / `product_id` | 输入行 | 身份，供后续对齐与去重 |
| `query` / `title` / `brand` | 输入行 | 原始文本，方便人工看错例 |
| `title_clipped` / `brand_clipped` | 输入行 | 截断诊断布尔值（**不参与模型输入**） |
| `gold` | 输入行的 `label` | 真实类别；F 的输入文件里没有 `label`，所以评测时会先拼进去（见 [04 §10](04-training.md)） |
| `score` | `Σ 概率 × GAINS` | 排序分：期望增益，范围 0–1 |
| `prediction` | 四类概率的最大值 | 分类结果 |
| `input_tokens` | `infer()` | 序列长度 |
| `probabilities` | softmax 结果 | `{'E':…, 'S':…, 'C':…, 'I':…}` |

`{**{...}, ...}` 是字典解包合并：先放身份与文本字段，再叠加其余字段。`dict(zip(LABELS, probabilities))` 把列表变成以标签为键的字典。

**`score` 手算示例**（真实数据，`example_id=1578411`）：

```
probabilities = {'E': 0.3247198760509491, 'S': 0.5437775254249573,
                 'C': 0.012219439260661602, 'I': 0.11928313970565796}
score = 0.3247198760509491×1 + 0.5437775254249573×0.1
      + 0.012219439260661602×0.01 + 0.11928313970565796×0
      = 0.3247198760509491 + 0.05437775254249573 + 0.00012219439260661602
      = 0.37921982298605145
```

和文件里的 `score: 0.37921982298605145` 完全一致。这个分数把“E 的概率”当主项，“S”打 1 折，“C”打 1%，所以它主要衡量“这个商品有多像完全满足需求的商品”。排序时用它，就得到 nDCG 里的那个顺序。

**依赖陷阱**：`predict()` 硬性要求输入行有 `title_clipped`、`brand_clipped` 两个键。当前的 `data/final/input.jsonl`、`data/sft/tune.jsonl` 都有；如果你换一份没有这两个字段的数据，会在这里 `KeyError`。

---

## 7. 损失由训练阶段负责

本模块不定义训练目标。SFT 使用真实标签 CE；Fusion 使用 `CE` 或 `(CE+KL)/2`，目标冻结且温度为 1。两阶段共用本文件的原始模型前向和保存实现。

---

## 8. 形状速查

| 阶段 | 张量 | 形状 | dtype |
| --- | --- | --- | --- |
| 单条编码 | `input_ids` / `attention_mask` | `[L]`（L≈131–245） | Python `int` 列表 |
| 组成批 | `input_ids` | `[B, Lmax]`（B=4） | `torch.long`, cuda |
| 主体输出 | `last_hidden_state` | `[B, Lmax, H]`（H=1024） | bf16 |
| 取末位 | `h` | `[B, H]` | fp32 |
| 标签权重 | `weights` | `[4, H]` | fp32 |
| logits | `forward()` 返回值 | `[B, 4]` | fp32 |
| 概率 | `softmax(-1)` | `[B, 4]` | fp32 |
| 损失 | `loss_rows()` 返回值 | `[B]` | fp32 |

---

## 9. 坑与细节

| 现象 / 写法 | 解释 |
| --- | --- |
| 配置里没有 `lora_r` 却引用 `config['lora_r']` | 该分支本轮不执行；`checkpoint` 总是存在 |
| `label_ids` 不在 `state_dict` 里 | `persistent=False`，按当前分词器重新计算 |
| 改了 `LABELS` 顺序，adapter 直接失效 | `label_ids` 与 `forward()` 的列顺序都跟着变，但权重是按旧顺序学的 |
| CPU 上构造模型可以，训练不行 | `train_arm()` 把 labels/targets 张量硬编码到 `'cuda'`，且 `gpu_required()` 会先拦截 |
| `input_tokens` 比商品标题长得多 | 它包含整条系统提示（约百 token） |
| `F` 在文件里既指 `torch.nn.functional` 又指评测集 | 本文件里一律是前者；文档里评测集写作“F 划分” |
| 推理时 `title_clipped` 缺失会 `KeyError` | `predict()` 要求这两个键存在 |

---

## 10. 自测

1. 如果把 `collate()` 改成左边补齐，`forward()` 里哪一行会算错？为什么？
2. `forward()` 返回的是 logits 还是概率？在哪里变成概率？
3. 为什么 `save()` 写的 `checkpoint_manifest.json` 不包含 `state.json`？
4. 评测时 `infer()` 已经 `model.eval()`，`train_arm()` 里为什么还要 `model.train()`？
5. 为什么 SFT 在训练循环中计算 CE，而续训使用 `loss_rows()`？这会改变共用的四分类前向吗？

参考思路：1 最后位置会落到 padding 上，`mask.sum()-1` 不再指向真实末尾；2 logits，`infer()` 的 `softmax(-1)`；3 因为 `state.json` 在 `snapshot()` 里后写，`save()` 执行时它还不存在；4 训练需要 dropout 处于训练模式（虽然本轮 dropout 被强制为 0，语义上仍应 `train()`）；5 会，损失尺度不同，等效学习率不同。

---

## 11. 验证

本文件导入了 `torch`，没有 GPU 环境的机器上不能直接跑。能做的静态检查：

```powershell
# 列出 modeling.py 里的类与函数（不需要 torch）
python -c "import ast; t=ast.parse(open('src/modeling.py',encoding='utf-8').read()); print([n.name for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])"

# 列出当前训练阶段的损失位置
Select-String -Path src\sft.py,src\fusion.py -Pattern 'cross_entropy|def loss_rows'
```

AST 列出的顶层定义为 `['RelevanceModel', 'infer', 'predict']`。所有模型函数保留原文；两个训练阶段通过同一个 `RelevanceModel.forward()` 取得四标签 logits。

有 GPU 与依赖时，最小验证是预检里的 `--gpu-check`（见[训练与评测](../训练与评测.md)）：它加载 S、对 8 条样本做前向，并检查概率有限。

下一站：[03 目标与损失](03-fusion-targets.md) —— 这些 logits 怎样变成三组不同的训练信号。
