# CyEDA BDD100k Day-to-Night

## 專案說明

使用 CyEDA（Cycle-object Edge Consistency Domain Adaptation）對 BDD100k 資料集進行日夜影像風格轉換訓練。
原始訓練流程來自 Google Colab Notebook，本專案將其改寫為可在本地環境執行的 Python 腳本。

- 原始論文：CyEDA: Cycle-object Edge Consistency Domain Adaptation（ICIP 2022）
- CyEDA 來源：https://github.com/bjc1999/CyEDA
- 資料集：BDD100k（日間 / 夜間子集）

## 目前狀態（2026-04-14）✅ 環境已完整建立，可直接訓練

所有前置作業均已完成：

| 項目 | 狀態 |
|---|---|
| `venv/`（Python 3.12 + PyTorch 2.6.0+cu124） | ✅ 已建立 |
| `CyEDA/`（訓練框架） | ✅ 已 clone |
| `bdd100k_dataset/trainA/`（1100 張日間訓練） | ✅ 已下載 |
| `bdd100k_dataset/trainB/`（1100 張夜間訓練） | ✅ 已下載 |
| `bdd100k_dataset/valA/`（6 張日間驗證） | ✅ 已下載 |
| `bdd100k_dataset/valB/`（6 張夜間驗證） | ✅ 已下載 |
| 最小批次測試（1 epoch, 10 張）| ✅ 通過（exit code 0） |

### 重啟 VS Code 後直接執行

```
venv\Scripts\activate
python CyEDA_BDD100k_D2N.py
```

## 目錄結構

```
CyEDA-BDD100k-D2N/
├── CyEDA_BDD100k_D2N.ipynb        # 原始 Colab Notebook
├── CyEDA_BDD100k_D2N.py           # 本地可執行訓練腳本（主程式）
├── requirements.txt                # Python 套件需求
├── CLAUDE.md                       # 本檔案
├── venv/                           # 虛擬環境（PyTorch 2.6.0+cu124）
├── CyEDA/                          # 訓練框架（已 clone）
│   ├── train.py                    # ⚠️ 已修改：testopt.nThreads = 0
│   ├── models/
│   ├── options/
│   ├── data/
│   └── checkpoints/test_run/       # 測試結果（已存在）
└── bdd100k_dataset/                # 資料集（已下載）
    ├── trainA/  (1100 images)
    ├── trainB/  (1100 images)
    ├── valA/    (6 images)
    └── valB/    (6 images)
```

## 環境規格

- GPU：NVIDIA GeForce RTX 3060（12GB VRAM）
- Driver：591.86，支援 CUDA 13.1
- Python：3.12.10
- PyTorch：2.6.0+cu124

### 安裝方式（已完成，紀錄用）

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install gdown opencv-python
```

## 訓練參數

| 參數 | 值 | 說明 |
|---|---|---|
| `--model` | `cycle_gan` | CycleGAN 模型 |
| `--dataset_mode` | `unaligned` | 不對齊的 A/B 資料集 |
| `--which_model_netG` | `sid_unet_resize` | U-Net Generator |
| `--which_model_netD` | `no_norm` | 無 BatchNorm Discriminator |
| `--no_dropout` | （flag） | Generator 不使用 Dropout |
| `--niter` | 20 | 固定學習率 epochs |
| `--niter_decay` | 20 | 學習率衰減 epochs |
| `--save_epoch_freq` | 1 | 每 epoch 存一次 checkpoint |
| `--pool_size` | 50 | 假影像緩衝池大小 |
| `--n_layers_D` | 5 | Discriminator 層數 |
| `--loadSize` | 286 | 載入後縮放尺寸（再裁至 fineSize） |
| `--fineSize` | 256 | 訓練影像尺寸 |
| `--resize_or_crop` | `resize` | 只做縮放，不做隨機裁剪 |
| `--no_flip` | （flag） | 不做水平翻轉資料增強 |
| `--tanh` | （flag） | 輸出層使用 tanh 激活 |
| `--batchSize` | 4 | RTX 3060 12GB 可安全使用 |
| `--gpu_ids` | 0 | 使用第 0 張 GPU |
| `--n_mask` | 3 | Edge loss 使用的 mask 數量 |
| `--lambda_cycle` | 3.0 | Cycle consistency 損失權重 |
| `--cycle_loss` | `Edge` | 使用 Edge loss |
| `--display_id` | 0 | 關閉 Visdom |
| `--nThreads` | 0 | Windows 必須設 0 |

## 已修改的 CyEDA 原始碼

| 檔案 | 行號 | 原內容 | 修改後 | 原因 |
|---|---|---|---|---|
| `CyEDA/train.py` | 12 | `testopt.nThreads = 1` | `testopt.nThreads = 0` | Windows multiprocessing spawn 不支援多執行緒 DataLoader |

## 訓練輸出

- Checkpoint 存於：`CyEDA/checkpoints/experimentA/`
- 驗證影像存於：`CyEDA/checkpoints/experimentA/web/images/`
- 格式：`epoch{NNN}_val.jpg`

## 注意事項

- 訓練前腳本會執行 `os.chdir(CYEDA_DIR)`，切換到 `CyEDA/` 子目錄再呼叫 `train.py`，因此所有相對路徑（`../bdd100k_dataset`、`./checkpoints`）均以 `CyEDA/` 為基準
- CPU 訓練可行但極慢，可將 `--gpu_ids` 改為 `-1`
- 訓練完成後 Step 6 會用 `cv2.imshow` 顯示 epoch 9 和 epoch 12 的驗證結果
- 完整訓練（40 epochs × 1100 張）預估需要數小時
