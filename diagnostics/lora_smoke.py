# -*- coding: utf-8 -*-
"""LoRA 微调冒烟验证：Qwen2.5-0.5B + peft，12 条门禁指令数据，CPU 1 epoch。

目的：验证教学文档第 05 章的 transformers+peft 代码路线在本机可行，
产出（训练时长/可训参数占比）回填教学文档。
用法：python diagnostics/lora_smoke.py
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = r"E:\datasets\models\qwen2.5-0.5b-instruct"
_OUT_DIR = r"E:\datasets\lora_smoke_out"
_DATA = os.path.join(os.path.dirname(_PROJECT_ROOT), "teaching_data",
                     "llm_finetune", "door_assistant_samples.jsonl")

import torch  # noqa: E402

from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from datasets import Dataset  # noqa: E402


def main():
    print("加载模型与分词器…")
    tok = AutoTokenizer.from_pretrained(_MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(_MODEL_DIR, torch_dtype=torch.float32)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 数据：Alpaca 三字段 → Qwen chat template → tokenize
    samples = []
    with open(_DATA, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            user_text = s["instruction"] + (f"\n{s['input']}" if s["input"] else "")
            msgs = [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": s["output"]}]
            samples.append({"text": tok.apply_chat_template(msgs, tokenize=False)})
    ds = Dataset.from_list(samples)

    def tokenize(ex):
        return tok(ex["text"], truncation=True, max_length=256)

    ds = ds.map(tokenize, remove_columns=["text"])

    t0 = time.time()
    args = TrainingArguments(
        output_dir=_OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    print(f"训练开始：{len(ds)} 条数据，1 epoch，CPU")
    trainer.train()
    dt = time.time() - t0
    print(f"\n训练完成，用时 {dt:.0f}s（{dt/60:.1f} 分钟）")

    # 合并 LoRA 并保存（验证 merge 流程）
    merged = model.merge_and_unload()
    merged.save_pretrained(os.path.join(_OUT_DIR, "merged"))
    tok.save_pretrained(os.path.join(_OUT_DIR, "merged"))
    print("合并权重已保存:", os.path.join(_OUT_DIR, "merged"))


if __name__ == "__main__":
    main()
