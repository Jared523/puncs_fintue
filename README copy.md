# 医疗标点恢复模型

`model.py`实现了Bert、BertLSTM、BertCRF三个模型。

### 预训练

使用了哈工大的[TextBrewer](https://github.com/airaria/TextBrewer)工具包实现知识蒸馏预训练过程，预训练使用了约两百万条中文医疗数据。

| 模型 | Batch Size | Training Steps | Learning Rate | Temperature | Teacher |
| :------- | :---------: | :---------: | :---------: | :---------: |  :---------: |
| **MiniRBT-H256** | 64 | 4万步 | 4e-4 | 8 | RoBERTa-wwm-ext |


将蒸馏后的模型（`distil`目录下）转成`pytorch_model.bin`，保存到`bert`路径下。
```
python convert.py
```

### 数据集
`data`下`train.json`、`dev.json`、`test.json`

```json
[
    {
        "tokens": ["体","格","检","查","舌","质","暗","红","舌","下","络","脉","瘀","紫","苔","薄","黄","脉","弦","细"],
        "tags": ["O","O","O","COMMA","O","O","O","COMMA","O","O","O","O","O","COMMA","O","O","COMMA","O","O","PERIOD"]
    }
]

```

句子恢复标点后：`体格检查，舌质暗红，舌下络脉瘀紫，苔薄黄，脉弦细。`

## Main APP
| Arguments                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| data_dir                | 数据集路径                                 |
| model_type             | 可选bert,lstm,crf    |
| model_name_or_path             | bert路径    |
| output_dir | 训练的模型存放路径 |
| data_cache_dir | 数据缓存路径|
| max_seq_length | 句子最大长度 |
| do_train | 训练 |
| do_eval | 测试 |
| logging_steps | 验证步数 |
| learning_rate | 学习率 |
| weight_decay | 权重衰减 |
| max_grad_norm | 正则最大范数 |
| batch_size | Batch size per GPU/CPU for training. |
| learning_rate | The initial learning rate for AdamW.|
| num_train_epochs | Total number of training epochs to perform. |
| seed | 随机数种子 |
| overwrite_cache | 重写数据缓存 |
| overwrite_output_dir | 重写训练的模型 |

