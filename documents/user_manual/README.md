# 用户手册
也许这个格式能变得再好看一点

## [Tutorial]()

## 用户API

### Config部分

### Data部分

### Model部分
    1） model = AbstractRecommender(config:Config, dataset:Dataset).to(device)
    将 AbstarctRecommender替换成想使用的model名称，初始化这个model
    参数：config(Config) -- config信息
         dataset(Dataset) -- dataset信息
    返回：

### Trainer部分
    1） trainer = Trainer(config:Config, model:AbstractRecommender, logger:Logger)
    声明一个trainer，控制模型训练和测试
    参数： config(Config) -- config信息
          model(AbstractRecommender) -- model
          logger(Logger) -- logger
    返回：
    
    2） trainer.fit(train_data:DataLoader, valid_data:DataLoader=None)
    输入训练数据和验证数据对模型开始训练，按照trainer接收的config信息进行训练，若输入valid_data, 会在验证集上最优时停止训练，
    若valid_data为空，按预先设定的epochs停止训练。
    参数：train_data(DataLoader) -- 训练数据
         valid_data(DataLoader) -- 验证数据
    返回：best_valid_score(float) -- 用于验证的最佳验证分数
         best_valid_result(dict) -- 用于验证的最佳验证结果
         
    3） trainer.evaluate(eval_data:DataLoader, load_best_model:bool=True, model_file:file=None)
    对输入的eval_data进行评测
    参数：train_data(DataLoader) -- 待验证数据
         load_best_model(bool) -- Trainer执行完fit方法后，当前model参数并不是最优，可以选择是否读取训练过程中最优的模型，默认True
         model_file(file) -- 若已经训练好模型，希望只调用Trainer的evaluate方法，输入含有模型参数的文件，读取其中的参数信息，此项优先级最高。       
    返回：result(dict) -- 评测结果
    
    4） trainer.resume_checkpoint(resume_file:file)
    加载文件中的模型信息及参数信息。trainer可以加载已经训练了一段时间但没训练完全的模型继续完成训练。
    参数：resume_file(file) -- 模型文件
    返回：

### Evaluate部分
