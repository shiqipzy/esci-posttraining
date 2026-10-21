"""Qwen four-label classification, LoRA adapters and training losses."""
from pathlib import Path
import torch
import torch.nn.functional as F
from common import GAINS, LABELS, collate, digest, encode, verify_model, write_json


class RelevanceModel(torch.nn.Module):
    def __init__(self, source, config, device='cuda', trainable=False, checkpoint=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.run_config = config
        self.provenance = verify_model(str(Path(source).resolve()))
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.amp_enabled = str(device).startswith('cuda')
        if self.amp_enabled and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('This frozen GPU recipe requires CUDA BF16. Do not silently switch precision.')
        dtype = torch.bfloat16 if self.amp_enabled else torch.float32
        from peft import LoraConfig, PeftModel, get_peft_model
        self.backbone = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=dtype, attn_implementation='sdpa')
        self.backbone.config.use_cache = False
        ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
        if not all(len(x) == 1 for x in ids):
            raise ValueError('Labels must be single tokens')
        self.register_buffer('label_ids', torch.tensor([x[0] for x in ids]), persistent=False)
        if checkpoint:
            self.backbone = PeftModel.from_pretrained(self.backbone, str(Path(checkpoint) / 'adapter'),
                                                      is_trainable=trainable)
        elif trainable:
            self.backbone = get_peft_model(self.backbone, LoraConfig(task_type='CAUSAL_LM',
                r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                target_modules=config['target_modules'], bias='none'))
        if trainable and config['gradient_checkpointing']:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.backbone.enable_input_require_grads()
        self.to(device)
        if not trainable:
            self.requires_grad_(False)
            self.eval()

    def base_model(self):
        # PeftModel wraps the Qwen model; a bare backbone is already the base model.
        getter = getattr(self.backbone, 'get_base_model', None)
        return getter() if getter is not None else self.backbone

    def forward(self, features):
        # LoRA modules are installed inside the underlying Qwen layers. Avoid materializing
        # [batch, sequence, vocabulary] logits; project only the last valid state to four labels.
        base = self.base_model()
        out = base.model(**features, use_cache=False).last_hidden_state
        last = features['attention_mask'].sum(dim=-1) - 1
        h = out[torch.arange(out.shape[0], device=out.device), last].float()
        weights = base.lm_head.weight.index_select(0, self.label_ids).float()
        return F.linear(h, weights)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(directory / 'adapter', safe_serialization=True)
        self.tokenizer.save_pretrained(directory / 'tokenizer')
        write_json(directory / 'run_config.json', self.run_config)
        write_json(directory / 'checkpoint_manifest.json', {'files': {
            str(p.relative_to(directory)).replace('\\', '/'): digest(p)
            for p in directory.rglob('*') if p.is_file()}})


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
