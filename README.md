# QwenASR-Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jRW4pDSLSK7JDNrgbfPi7sKHVzlYZ4y8)

在免費 Colab（T4）上跑 **Qwen3-ASR-1.7B**：上傳音檔 → 轉錄 → 下載 SRT。不用本機 GPU，不做量化（fp16）。

## English TL;DR

Run Qwen3-ASR-1.7B on free Colab (T4): upload audio → transcribe → download SRT. No local GPU needed, no quantization (fp16). The real value of this repo is the [pitfalls section](#踩過的坑v2v9-血淚史) — every entry below cost a real failed run.

## 怎麼用

1. 用 Colab 開啟 `QwenASR-Colab.ipynb`（建議「檔案 → 在雲端硬碟中儲存複本」）
2. 選 GPU runtime（免費 T4 即可）
3. 由上往下依序執行；到上傳格時上傳音檔（mp3 / wav / flac / m4a / ogg / mp4 / mkv 皆可，影片會自動抽音軌）
4. 跑完自動下載 `.srt`

> 免費版 Colab session 上限約 12 小時，閒置會斷線。斷線後重跑整份 notebook 即可（模型要重下，約 4.7GB，幾分鐘）。

## 實測數據（2026-09-27，免費 Colab，真實跑過）

- GPU：Tesla T4 / 14.6GB VRAM / torch 2.11.0+cu128 / cuda 12.8
- 精度：float16（T4 不支援 bf16，見坑 #6）
- 主模型載入：約 135 秒；RAM 峰值約 5GB / 12.67GB；VRAM 6.1GB
- 對齊模型（Qwen3-ForcedAligner-0.6B，字級時間軸用）：約 22 秒
- 4.2 秒中文測試音檔：VAD 切出 1 段，轉錄 3.4 秒，產出 1 行字幕

## 踩過的坑（v2→v9 血淚史）

### 1. 免費 Colab 的 RAM 只有 12.67GB：`from_pretrained()` 會 OOM

`Qwen3ASRModel.from_pretrained()` 在 CPU staging 階段就會吃掉超過 11GB RAM，「主模型就緒」還沒印出來就 OOM（v2、v3 連續陣亡）。`low_cpu_mem_usage=True` 也救不了。

**解法**：`accelerate.init_empty_weights()` 建 meta 空殼，再逐 tensor 把權重直上 GPU，RAM 峰值壓到約 5GB。

### 2. 選錯 repo：`-hf` 版鍵名對不上

`Qwen/Qwen3-ASR-1.7B-hf` 的權重鍵是 `model.*`，qwen-asr 0.0.6 的模型類別要的是 `thinker.*`——對著 `-hf` 版怎麼載鍵名都對不上（v4、v5 陣亡）。**要用原版 `Qwen/Qwen3-ASR-1.7B`（無 `-hf`）**。

### 3. 上游 bug：`rope_scaling=None`

官方 config 的 `rope_scaling` 是 `None`，但 vendored 的 modeling 直接 `.get()`，炸 `AttributeError: 'NoneType' object has no attribute 'get'`。

**Workaround**（notebook 內已含）：建模型前直接把 `config.rope_scaling` 補上預設值 `{"rope_type": "default", "mrope_section": [24, 20, 20]}`。注意：舊版曾用 monkeypatch 去包 `__init__`，但那格 cell 重跑第二次會把 patch 疊在 patch 上、造成 `RecursionError`（2026-09-27 實測踩到），已改為冪等的寫法。

### 4. `load_state_dict` / `safetensors.load_model` 載不進 meta 模型

這兩個都只是把值 copy 進「已存在的 storage」，而 meta tensor 根本沒有 storage——實測 708 個參數**全部**殘留 meta，載了等於沒載（v6、v7 陣亡）。

**正解**：`accelerate.utils.set_module_tensor_to_device` 逐 tensor 取代參數物件（這也是 accelerate 自己載入檢查點用的機制）。

### 5. non-persistent buffers 留在 CPU

`init_empty_weights` 預設只把 parameter 放 meta，non-persistent buffers（如 `SinusoidsPositionEmbedding`、rotary `inv_freq`）留在 CPU。前向傳播時 CUDA 的 tensor ＋ CPU 的 buffer 相加 → `RuntimeError`（v8 陣亡，位置在 audio tower 的 positional embedding 相加那行）。

**解法**：載完後把所有 buffer 搬上 cuda 並轉目標精度，最後用斷言確認全模型無 meta 且全在 cuda 上。

### 6. 精度選擇不要用猜的

T4 不支援 bf16。用 `torch.cuda.is_bf16_supported()` 判斷，支援才用 bf16，否則 fp16。不要假設「非 T4 就支援 bf16」。

## 網路行為

- pip 安裝套件（PyPI）
- 從 Hugging Face 下載模型權重（主模型 `Qwen/Qwen3-ASR-1.7B` 約 4.7GB；Forced Aligner `Qwen/Qwen3-ForcedAligner-0.6B` 另需約 1.2GB）
- 不上傳任何資料到任何地方；轉錄全在 Colab VM 本機完成

## 授權與維護狀態

MIT License。Verified on 2026-09-27；**不承諾維護**——上游（qwen-asr / transformers）版本漂移後請自行驗證。
