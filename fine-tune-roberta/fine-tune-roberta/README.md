# HOWTO

## `run.py --data_path 训练数据路径 --model_path roberta模型路径 --pred_data 预测json文件路径 `

## 跑完后会生成`output.json`，拿它去跑`python evaluation_script.py /path_to/official_dev.json /path_to/output.json `。此外，还会保存微调好的两个roberta，一个answer.model，一个evidence.model。

