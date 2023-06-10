# SAM-3D



Train:

```bash
python train.py 
		--task_name your_task_name 
		--click_type random  # 取点方式
		--multi_click  # 是否是多个点击
		--gpu_ids 0 1 
		--multi_gpu  # 是否多卡训练

```





Inference:

``` bash
python inference3D_all.py 
		-cp your_weight_path  # 训练好的权重路径
		-dt Va 
		-pm random 
		--multi 
		--union 
		-tdp your_validation_data_path  # 测试数据路径 
```















