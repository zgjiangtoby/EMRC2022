# HOWTO

1. 训练模式：`python run.py --data_path 微调数据路径 --ans_model dir_to_/chinese-roberta-wwm-ext-large --evi_model dir_to_/chinese-roberta-wwm-ext-large --pred_data dir_to/expmrc-cmrc2018-dev.json --train True --predict False`

2. 预测模式：`python run.py --data_path 微调数据路径 --ans_model dir_to_/ans_model --evi_model dir_to_/evi_model --pred_data dir_to/expmrc-cmrc2018-dev.json --train False --predict True`

3. 跑完后会生成 `\results`路径，并产生`output.json`，拿它去跑`python evaluation_script.py /path_to/official_dev.json /path_to/output.json `。此外，还会保存微调好的两个roberta，一个answer.model，一个evidence.model。

### 注：如果没有gpu，可以将`run.py`中，将`training_args`中的no_cuda改为True，就可以在CPU上跑了。
