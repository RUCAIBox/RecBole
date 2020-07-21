# 用户手册
也许这个格式能变得再好看一点

## [Tutorial]()

## 用户API

### Config部分

### Data部分

### Model部分
1) `model = AbstractRecommender(config:Config, dataset:Dataset)`

   将 `AbstarctRecommender`替换成想使用的model名称，初始化这个model 

   参数：
   - `config(Config)` config对象
   - `dataset(Dataset)` dataset对象

   返回：

### Trainer部分
1） `trainer = Trainer(config:Config, model:AbstractRecommender, logger:Logger)`
   
   声明一个`trainer`，控制模型训练和测试

   参数： 
   - `config(Config)` config对象
   - `model(AbstractRecommender)` model对象
   - `logger(Logger)` logger对象
   
   返回：
    
    2） trainer.fit(train_data:DataLoader, valid_data:DataLoader=None)
    输入训练数据和验证数据对模型开始训练，按照trainer接收的config信息进行训练，若输入valid_data, 会在验证集上最优时停止训练，
    若valid_data为空，按预先设定的epochs停止训练。
    参数：train_data(DataLoader) -- 训练数据
         valid_data(DataLoader) -- 验证数据
    返回：best_valid_score(float) -- 用于验证的最佳验证分数
         best_valid_result(dict) -- 用于验证的最佳验证结果
         
    3） trainer.evaluate(eval_data:DataLoader, load_best_model:bool=True, model_file:str=None)
    对输入的eval_data进行评测
    参数：train_data(DataLoader) -- 待验证数据
         load_best_model(bool) -- Trainer执行完fit方法后，当前model参数并不是最优，可以选择是否读取训练过程中最优的模型，默认True
         model_file(str) -- 若已经训练好模型，希望只调用Trainer的evaluate方法，输入含有模型参数的文件，读取其中的参数信息，此项优先级最高。       
    返回：result(dict) -- 评测结果
    
    4） trainer.resume_checkpoint(resume_file:str)
    加载文件中的模型信息及参数信息。trainer可以加载已经训练了一段时间但没训练完全的模型继续完成训练。
    参数：resume_file(str) -- 模型文件
    返回：
    
    5） hp = HyperTuning(procedure_file:str, space:dict=None, params_file:str=None, interpreter:str='python', 
                        algo:hyperopt.tpe=hyperopt.tpe.suggest, max_evals:int=100, bigger:bool=True)
    实例化HyperTuning的一个对象hp，hp用于控制输入的python文件，自动调整超参数寻找最优
    参数：procedure_file(str) -- hp调用的程序文件 *.py ，该程序文件需要包含完整的训练过程，保证可以正确单独运行完成训练，
         并将best_valid_score（验证集合上的最佳分数）打印出来。参考文件`run_test.py`。
         space(dict) -- 要调整的超参数字典，其中key为超参数名称，value为hyperopt类型的参数范围，由于本模块为基于hyperopt的进一步封装，
                        space具体可以参考hyperopt这个库：https://github.com/hyperopt/hyperopt
                        一个例子：
                            space = {
                                'train_batch_size': hyperopt.hp.choice('train_batch_size',[512, 1024, 2048])
                                'learning_rate': hp.loguniform('learning_rate', -8, 0)
                            }
         params_file(str) -- 记录要调整的超参数信息的文件，为了那些无需过多复杂超参数设置的用户或想便捷使用超参调整模块的用户准备。
                             将要调整的超参简单写入文本，我们将其转换为space变量，如果space变量不为空，这个参数将没有意义，
                             用户需要保证space和params_file至少一个不为空。
                             params_file的格式要求如下：
                             一行对应一个参数，具体包括参数名称，搜索类型，搜索范围，中间用' '隔开。
                             由于是简化版，变化类别只能支持四类，但足够覆盖正常的模型超参数，若想使用更复杂的参数变化类别，请设置space参数
                             搜索类型：choice， 搜索范围：options(list)，说明：从options中的元素搜索
                             搜索类型：uniform，搜索范围：low(int),high(int),说明：从(low,high)的均匀分布中搜索
                             搜索类型：loguniform, 搜索范围：low(int),high(int),说明：从exp(uniform(low,high))的均匀分布中搜索
                             搜索类型：quniform，搜索范围：low(int),high(int),q(int)，说明：从round(uniform(low, high) / q) * q的均匀分布搜索 
                             一个例子：'hyper.test'
                             了解更多信息请参考：https://github.com/hyperopt/hyperopt/wiki/FMin#2-defining-a-search-space
         interpreter(str) -- 执行procedure_file所用的解释器，默认'python'
         algo() -- 优化算法，默认hyper.tpe.suggest
         max_evals(int) -- 最大搜索次数，默认100次
         bigger(bool) -- procedure_file中的best_valid_score是否是越大越好，默认True
    返回：
    
    6) hp.run()
    开始自动调整超参数
    

### Evaluate部分
