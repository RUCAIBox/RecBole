# RecBole Docker 故障排除指南

這個指南基於實際修復經驗，涵蓋了在 Python 3.10 環境中運行 RecBole 時可能遇到的所有問題和解決方案。

## 🎯 已修復的兼容性問題

這個 Docker 環境已經修復了以下主要兼容性問題：

### **核心庫兼容性**
- ✅ **numpy 2.0 兼容性**：修復了 `np.float_`, `np.int_`, `np.bool_`, `np.long_` 等過時屬性
- ✅ **scipy 兼容性**：修復了 `dok_matrix._update()` 方法被移除的問題
- ✅ **PyTorch 2.6 兼容性**：修復了 `torch.load()` 的 `weights_only` 參數問題
- ✅ **ray tune 延遲導入**：解決了不需要 hyperparameter tuning 時的導入錯誤

### **受影響的文件和模型**

**numpy 2.0 修復（4 個文件）：**
- `recbole/config/configurator.py` - 核心兼容性設置
- `recbole/evaluator/metrics.py` - 評估指標計算
- `recbole/trainer/hyper_tuning.py` - 超參數調整
- `recbole/model/layers.py` - 模型層定義

**scipy ._update 修復（7 個模型）：**
- `NGCF`, `LightGCN`, `NCL`, `GCMC`, `SpectralCF`, `KGIN`, `MCCLK` 等圖神經網路模型

**torch.load 修復（8 個文件）：**
- `recbole/trainer/trainer.py` - 模型訓練器
- `recbole/quick_start/quick_start.py` - 快速開始工具
- `S3Rec`, `RACT`, `NeuMF`, `NAIS`, `ConvNCF`, `KD_DAGFM` 等預訓練模型

### **測試結果確認**
✅ **模型讀取機制**：`load_data_and_model()` 函數正常運行
✅ **推論功能**：包括 embedding 提取、評分預測、推薦排序、批次推論
✅ **評估機制**：RecBole 內建評估器正常計算所有指標

## 🔧 常見問題和解決方案

### 1. **Docker 重建相關問題**

#### ❓ 什麼時候需要重新 build Docker？

**通常不需要！** 因為我們使用了 volume mount (`-v .:/app`) 和開發模式安裝 (`pip install -e .`)

**只有在以下情況才需要重新 build：**
```bash
# 修改了 requirements.txt
# 修改了 Dockerfile
# 修改了系統層面的依賴

docker-compose build recbole
```

**修改 RecBole 源碼時：**
```bash
# ✅ 不需要重新 build，修改立即生效
# 直接編輯 recbole/ 目錄下的 .py 檔案
# 修改會立即在容器中生效
```

### 2. **進入容器後的基本檢查**

```bash
# 檢查環境
python test_environment.py

# 檢查當前位置
pwd  # 應該顯示 /app

# 檢查 Python 版本
python --version  # 應該是 Python 3.10.x

# 檢查 RecBole 是否正確安裝
python -c "import recbole; print(recbole.__version__)"
```

### 3. **模型配置相關問題**

#### 問題：`train_neg_sample_args should be None when the loss_type is CE`

**常見於：** SASRec, BERT4Rec 等序列推薦模型

```bash
# ❌ 錯誤的配置
run_recbole(model='SASRec', dataset='ml-100k')

# ✅ 正確的配置
run_recbole(
    model='SASRec', 
    dataset='ml-100k',
    config_dict={'train_neg_sample_args': None}
)
```

#### 問題：模型參數不兼容

```bash
# 檢查模型的預設配置
python -c "
from recbole.config import Config
config = Config(model='SASRec', dataset='ml-100k')
print('MODEL_TYPE:', config['MODEL_TYPE'])
print('MODEL_INPUT_TYPE:', config['MODEL_INPUT_TYPE'])
print('loss_type:', config.get('loss_type', 'N/A'))
"
```

### 4. **可選依賴問題**

#### 問題：`ModuleNotFoundError: No module named 'lightgbm'`

**影響模型：** 某些需要外部機器學習庫的模型

```bash
# 檢查是否需要額外依賴
pip install lightgbm

# 或者使用不需要額外依賴的模型
python -c "
from recbole.quick_start import run_recbole
# 使用核心模型，如 BPR, NeuMF, NGCF 等
run_recbole(model='BPR', dataset='ml-100k')
"
```

### 5. **資料集相關問題**

#### 問題：`Dataset [your_dataset] not found`

```bash
# 檢查資料集目錄
ls dataset/

# 使用現有資料集進行測試
python -c "
from recbole.quick_start import run_recbole
run_recbole(model='NGCF', dataset='ml-100k')
"
```

#### 解決方案：建立自訂資料集

```bash
# 建立資料集目錄
mkdir -p dataset/your_dataset

# 建立互動檔案（必需）
cat > dataset/your_dataset/your_dataset.inter << EOF
user_id:token	item_id:token	rating:float	timestamp:float
user1	item1	5.0	1234567890
user1	item2	4.0	1234567891
user2	item1	3.0	1234567892
EOF

# 建立物品檔案（可選）
cat > dataset/your_dataset/your_dataset.item << EOF
item_id:token	feature1:float	feature2:token
item1	0.5	category_A
item2	0.8	category_B
EOF
```

### 6. **記憶體和性能問題**

#### 問題：記憶體不足或訓練太慢

```bash
# 使用 CPU 並降低 batch size
python -c "
from recbole.quick_start import run_recbole
run_recbole(
    model='NGCF', 
    dataset='ml-100k',
    config_dict={
        'device': 'cpu', 
        'train_batch_size': 256,
        'epochs': 5  # 快速測試
    }
)
"
```

#### 問題：訓練時間太長（僅用於測試）

```bash
# 快速測試配置
config_dict = {
    'epochs': 1,
    'eval_step': 1,
    'train_batch_size': 512,
    'eval_batch_size': 1024
}
```

### 7. **版本兼容性問題（已修復但供參考）**

#### 如果遇到 numpy 2.0 相關錯誤

```bash
# 這些問題在我們的環境中已經修復，但如果在其他環境中遇到：
# AttributeError: module 'numpy' has no attribute 'float_'
# AttributeError: module 'numpy' has no attribute 'bool8'

# 解決方案已經整合在我們的 configurator.py 中
```

#### 如果遇到 scipy 相關錯誤

```bash
# AttributeError: 'dok_matrix' object has no attribute '_update'
# 解決方案已經整合在我們的 ngcf.py 中
```

### 8. **模型測試建議**

#### 推薦的測試順序

```bash
# 1. 基礎模型測試
python -c "
from recbole.quick_start import run_recbole
run_recbole(model='BPR', dataset='ml-100k', config_dict={'epochs': 1})
"

# 2. 神經網絡模型測試
python -c "
from recbole.quick_start import run_recbole
run_recbole(model='NeuMF', dataset='ml-100k', config_dict={'epochs': 1})
"

# 3. 圖神經網絡模型測試
python -c "
from recbole.quick_start import run_recbole
run_recbole(model='NGCF', dataset='ml-100k', config_dict={'epochs': 1})
"

# 4. 序列推薦模型測試
python -c "
from recbole.quick_start import run_recbole
run_recbole(model='SASRec', dataset='ml-100k', config_dict={
    'epochs': 1, 
    'train_neg_sample_args': None
})
"
```

### 9. **權限和檔案系統問題**

#### 問題：無法寫入檔案

```bash
# 檢查權限
ls -la /app/saved
ls -la /app/logs

# 修正權限（在容器內）
chmod 755 /app/saved /app/logs

# 如果在主機上修正權限
sudo chown -R $(id -u):$(id -g) saved/ logs/
```

### 10. **完整重置流程**

如果遇到無法解決的問題：

```bash
# 1. 退出容器
exit

# 2. 停止所有服務
docker-compose down

# 3. 刪除舊的鏡像和容器（謹慎使用）
docker-compose down --rmi all --volumes

# 4. 清理 Docker 系統（釋放空間）
docker system prune -f

# 5. 重新建立（無快取）
docker-compose build --no-cache recbole

# 6. 重新進入
docker-compose run --rm recbole bash
```

## 🚨 緊急除錯命令

```bash
# 快速環境診斷
python test_environment.py

# 詳細系統資訊
python -c "
import sys, torch, recbole, numpy, scipy
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'RecBole: {recbole.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# 檢查系統資源
free -h  # 記憶體
df -h    # 磁碟空間
```

## 🎯 不同模型類型的快速測試

```bash
# 基礎推薦模型
docker-compose run --rm recbole python -c "
from recbole.quick_start import run_recbole
run_recbole(model='BPR', dataset='ml-100k', config_dict={'epochs': 1})
"

# 深度學習模型
docker-compose run --rm recbole python -c "
from recbole.quick_start import run_recbole
run_recbole(model='NeuMF', dataset='ml-100k', config_dict={'epochs': 1})
"

# 圖神經網絡
docker-compose run --rm recbole python -c "
from recbole.quick_start import run_recbole
run_recbole(model='NGCF', dataset='ml-100k', config_dict={'epochs': 1})
"

# 序列推薦
docker-compose run --rm recbole python -c "
from recbole.quick_start import run_recbole
run_recbole(model='SASRec', dataset='ml-100k', config_dict={
    'epochs': 1, 'train_neg_sample_args': None
})
"
```

## 💡 最佳實踐

### 開發工作流程

1. **環境檢查**：`python test_environment.py`
2. **快速測試**：先用 1 epoch 測試模型是否能運行
3. **逐步驗證**：從簡單模型到複雜模型
4. **配置調整**：根據需要調整 batch size 和 epochs
5. **結果保存**：使用 volume mount 確保結果持久化

### 避免常見錯誤

1. **不要盲目增加 epochs**：先確保模型能正常運行
2. **注意模型特定配置**：每個模型可能有特殊要求
3. **檢查資料集格式**：確保資料集檔案格式正確
4. **監控系統資源**：避免記憶體耗盡

### 效能優化

```bash
# CPU 最佳化配置
config_dict = {
    'device': 'cpu',
    'train_batch_size': 512,
    'eval_batch_size': 1024,
    'worker': 4
}

# 快速測試配置
config_dict = {
    'epochs': 1,
    'eval_step': 1,
    'show_progress': False
}
```

## 📞 求助資源

如果以上方法都無法解決問題：

1. **檢查錯誤訊息**：提供完整的 traceback
2. **確認環境資訊**：Python、RecBole、依賴版本
3. **描述重現步驟**：包括使用的模型、資料集、配置
4. **查看官方資源**：
   - RecBole 官方文檔
   - GitHub Issues
   - 學術論文和範例

## 🔍 除錯檢查清單

遇到問題時，按照以下順序檢查：

- [ ] 執行 `python test_environment.py` 確認基礎環境
- [ ] 確認是否需要重新 build Docker（通常不需要）
- [ ] 檢查模型是否需要特殊配置（如 SASRec）
- [ ] 確認資料集檔案存在且格式正確
- [ ] 檢查是否有足夠的系統資源
- [ ] 嘗試使用更簡單的配置（更少 epochs、更小 batch size）
- [ ] 查看完整的錯誤訊息，不要只看最後一行

記住：這個環境已經解決了大部分兼容性問題，大多數問題都是配置相關的！🚀