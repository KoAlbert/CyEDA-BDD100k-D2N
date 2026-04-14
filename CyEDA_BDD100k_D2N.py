import os
import shutil
import subprocess
import sys
import zipfile

# ──────────────────────────────────────────────
# 工作目錄：腳本所在資料夾
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ──────────────────────────────────────────────
# GPU 確認（RTX 3060，CUDA）
# 若顯示 False 請先安裝 CUDA 版 PyTorch：
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# ──────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
else:
    print("[警告] CUDA 不可用，將使用 CPU 訓練（速度極慢）")
    print("       請確認已安裝 CUDA 版 PyTorch")

# ──────────────────────────────────────────────
# Step 1: 下載 BDD100k 日夜子集（Google Drive）
# update gdown for compatibility issue. 2022/12/12
# 若 trainA/trainB/valA/valB 已存在則略過
# ──────────────────────────────────────────────
DATASET_DIR = os.path.join(SCRIPT_DIR, "bdd100k_dataset")
DATASET_READY = all(
    os.path.isdir(os.path.join(DATASET_DIR, d))
    for d in ["trainA", "trainB", "valA", "valB"]
)

if not DATASET_READY:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"], check=True)
    import gdown

    #day train:
    #https://drive.google.com/file/d/1vTkyoQkJ13TuufkjO98Pg4q-aw2Zx5N5/view?usp=share_link
    #day val:
    #https://drive.google.com/file/d/1LxQ9LTaaHBdMkO9hY-mdvYATf9_J9olx/view?usp=share_link
    #night train:
    #https://drive.google.com/file/d/1pArk8JUYPc520Oalm7bMhprIxBu3IRou/view?usp=share_link
    #night val:
    #https://drive.google.com/file/d/1BV5ez0EhC8xjP329vmyKLmukdZk3kgfU/view?usp=share_link

    files = {
        "CycleEDA_bdd100k_day_train.zip":   "1vTkyoQkJ13TuufkjO98Pg4q-aw2Zx5N5",
        "CycleEDA_bdd100k_day_val.zip":     "1LxQ9LTaaHBdMkO9hY-mdvYATf9_J9olx",
        "CycleEDA_bdd100k_night_train.zip": "1pArk8JUYPc520Oalm7bMhprIxBu3IRou",
        "CycleEDA_bdd100k_night_val.zip":   "1BV5ez0EhC8xjP329vmyKLmukdZk3kgfU",
    }
    for fname, fid in files.items():
        if not os.path.exists(fname):
            gdown.download(f"https://drive.google.com/uc?id={fid}", fname, quiet=False)

    # ──────────────────────────────────────────────
    # Step 2: 解壓縮
    # ──────────────────────────────────────────────
    for zip_name in files.keys():
        with zipfile.ZipFile(zip_name, "r") as z:
            z.extractall(".")
        os.remove(zip_name)

    # ──────────────────────────────────────────────
    # Step 3: 整理成 trainA / trainB / valA / valB 結構
    # ──────────────────────────────────────────────
    os.makedirs(DATASET_DIR, exist_ok=True)
    #os.makedirs('bdd100k_dataset/valAB') # later be used for predict.sh

    shutil.move("./CycleEDA_bdd100k_day_train",   DATASET_DIR)
    shutil.move("./CycleEDA_bdd100k_night_train",  DATASET_DIR)
    shutil.move("./CycleEDA_bdd100k_day_val",      DATASET_DIR)
    shutil.move("./CycleEDA_bdd100k_night_val",    DATASET_DIR)

    os.rename(f"{DATASET_DIR}/CycleEDA_bdd100k_day_train",   f"{DATASET_DIR}/trainA")
    os.rename(f"{DATASET_DIR}/CycleEDA_bdd100k_night_train", f"{DATASET_DIR}/trainB")
    os.rename(f"{DATASET_DIR}/CycleEDA_bdd100k_day_val",     f"{DATASET_DIR}/valA")
    os.rename(f"{DATASET_DIR}/CycleEDA_bdd100k_night_val",   f"{DATASET_DIR}/valB")
else:
    print(f"[資料集] 已就緒：{DATASET_DIR}")

# ──────────────────────────────────────────────
# Step 4: Clone CyEDA 訓練框架
# 若目錄已存在則略過
# ──────────────────────────────────────────────
CYEDA_DIR = os.path.join(SCRIPT_DIR, "CyEDA")
if not os.path.exists(CYEDA_DIR):
    subprocess.run(["git", "clone", "https://github.com/bjc1999/CyEDA.git"], check=True)
else:
    print(f"[CyEDA] 已就緒：{CYEDA_DIR}")

os.chdir(CYEDA_DIR)
print("CyEDA 目錄內容：", os.listdir("."))

# ──────────────────────────────────────────────
# Step 5: 執行訓練
# 對應 train.sh 的參數設定
# --display_id 0：關閉 Visdom（本地不需要 web 視覺化伺服器）
# RTX 3060 12GB，batchSize 4 可安全使用
# ──────────────────────────────────────────────
train_cmd = [
    sys.executable, "train.py",
    "--dataroot",            "../bdd100k_dataset",
    "--checkpoints_dir",     "./checkpoints",
    "--no_dropout",
    "--name",                "experimentA",
    "--model",               "cycle_gan",
    "--dataset_mode",        "unaligned",
    "--which_model_netG",    "sid_unet_resize",
    "--which_model_netD",    "no_norm",
    "--save_epoch_freq",     "1",
    "--niter",               "40",
    "--niter_decay",         "20",
    "--pool_size",           "50",
    "--n_layers_D",          "5",
    "--loadSize",            "286",
    "--fineSize",            "256",
    "--resize_or_crop",      "resize",
    "--batchSize",           "4",  # RTX 3060 12GB，256x256 影像可安全使用 4
    "--no_flip",
    "--tanh",
    "--gpu_ids",             "0",
    "--n_mask",              "3",
    "--lambda_cycle",        "3.0",
    "--cycle_loss",          "Edge",
    "--display_id",          "0",  # 關閉 Visdom，本地訓練不需要 web 視覺化
    "--nThreads",            "0",  # Windows multiprocessing spawn 限制，必須設 0
]

subprocess.run(train_cmd, check=True)

# ──────────────────────────────────────────────
# Step 6: 顯示驗證結果影像
# Let's show some validation images.
# Do remember the result deviation is quite significant from epoch to epoch!
# ──────────────────────────────────────────────
import cv2

def show_image(path, title=""):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[警告] 找不到影像：{path}")
        return
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("result of epoch-9")
show_image("./checkpoints/experimentA/web/images/epoch009_val.jpg", "epoch-9")

print("result of epoch-12")
show_image("./checkpoints/experimentA/web/images/epoch012_val.jpg", "epoch-12")
