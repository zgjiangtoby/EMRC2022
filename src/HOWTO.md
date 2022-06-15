## How to

1. conda环境qust.yml
2. 创建一个preprocessed_data路径，然后使用`python preprocessing.py --data_path 你的路径/xxx.json --model_path 你的路径/chinese-roberta-wwm-ext-large --output 你的路径/preprocessed_data`
3. 模型在`model.py`,一个简单的cross-attention(协同注意力)，将自注意力的q在qustion和context之间互换。
4. 模型训练：`python run_big.py --data_path 你的路径/preprocessed_data/ --tokenizer_path 你的路径/chinese-roberta-wwm-ext-large`。
5. 模型调参： `/config/config.py`
6. 模型训练完成后会在当前路径保存训练好的模型`QA_1.model`
7. 模型预测（生成官方文档中要求的json文件）： `python run_big.py --predict True --model_path 你的路径/QA_1.model --pred_path 你的路径/expmrc-cmrc2018-dev.json`
8. 运行官方测试文件 `python evaluation_script.py 你的路径/expmrc-cmrc2018-dev.json 你的路径/model_outputs.json`



