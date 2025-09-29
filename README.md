
# M5Stack MLOps - Environmental Comfort Prediction System

M5StackとENV IIIセンサーを使用した環境快適度予測システムです。TensorFlow Liteを使用してエッジデバイスで機械学習モデルを実行し、温度・湿度・気圧から不快指数（THI）を予測します。

## システム概要

- **センサー**: SHT3X (温湿度), QMP6988 (気圧)
- **デバイス**: M5Stack Core
- **ML Framework**: TensorFlow Lite for Microcontrollers
- **データ保存**: Ambient IoTクラウド
- **モデル更新**: OTA (Over-The-Air) via WiFi

## 必要なハードウェア

- M5Stack Core (ESP32ベース)
- ENV III Unit (SHT3X + QMP6988)
- USB Type-Cケーブル

## 必要なソフトウェア

### Arduino環境
- Arduino IDE 1.8.x または 2.x
- ESP32ボードマネージャー

### Python環境
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn ambient requests schedule
```

---

## セットアップ手順

### 1. Arduino環境のセットアップ

#### 1.1 ボードマネージャーの設定
Arduino IDEで以下を設定:
1. `ファイル` → `環境設定`
2. 追加のボードマネージャーURL: 
   ```
   https://dl.espressif.com/dl/package_esp32_index.json
   ```
3. `ツール` → `ボード` → `ボードマネージャー`
4. "ESP32" を検索してインストール

#### 1.2 必要なライブラリのインストール
`スケッチ` → `ライブラリをインクルード` → `ライブラリを管理` から以下をインストール:

| ライブラリ名 | バージョン | 説明 |
|------------|----------|------|
| M5Stack | 最新 | M5Stack基本機能 |
| M5Unit-ENV | 最新 | ENV IIIセンサードライバ |
| Ambient ESP32 ESP8266 lib | 最新 | Ambientクラウド連携 |
| TensorFlowLite_ESP32 | 最新 | TensorFlow Lite推論エンジン |

**重要**: `TensorFlowLite_ESP32`は以下のURLから手動でインストールが必要な場合があります:
```
https://github.com/tanakamasayuki/TensorFlowLite_ESP32
```

#### 1.3 必要なヘッダーファイル

`MLOPS_M5STACK.ino`と同じディレクトリに以下のファイルが必要です:

1. **model.h** - 初期モデルデータ
   - TensorFlow Liteモデルのバイト配列
   - Pythonスクリプトで生成されたモデルをC配列に変換
   
2. **config.h** (作成が必要) - 認証情報
   ```cpp
   #ifndef CONFIG_H
   #define CONFIG_H
   
   // WiFi設定
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   
   // Ambient設定
   unsigned int channelId = YOUR_CHANNEL_ID;
   const char* writeKey = "YOUR_WRITE_KEY";
   
   #endif
   ```

#### 1.4 config.hの作成

```bash
# プロジェクトディレクトリで
cp config.h.example config.h
# config.hを編集して実際の認証情報を入力
```

### 2. M5Stackへの書き込み

1. M5StackをUSBで接続
2. `ツール` → `ボード` → `M5Stack-Core-ESP32`を選択
3. `ツール` → `シリアルポート` → 接続されたポートを選択
4. `ツール` → `Upload Speed` → `115200`を選択
5. `スケッチ` → `マイコンボードに書き込む`

**初回書き込み時の注意**:
- メモリ不足エラーが出る場合: `ツール` → `Partition Scheme` → `No OTA (Large APP)`を選択

---

## Python学習スクリプトのセットアップ

### 1. 設定ファイルの作成

`pyファイル`の32行目を編集:

```python
# Ambientクライアントの初期化
am = ambient.Ambient(
    "YOUR_CHANNEL_ID", 
    "YOUR_WRITE_KEY", 
    "YOUR_READ_KEY", 
    "YOUR_USER_KEY"
)
```

### 2. 実行方法

#### モード1: 学習とアップロード (推奨)
```bash
python MLops_training_tflite_uploadM5.py --mode both --ip 192.168.1.XXX
```

#### モード2: 学習のみ
```bash
python MLops_training_tflite_uploadM5.py --mode train
```

#### モード3: アップロードのみ
```bash
python MLops_training_tflite_uploadM5.py --mode upload --ip 192.168.1.XXX
```

#### 定期実行の設定
```bash
# 毎日午前3時に自動学習・アップロード
python MLops_training_tflite_uploadM5.py --mode both --ip 192.168.1.XXX --time 03:00
```

### 3. オプション

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|-------|------|-----------|
| `--mode` | `-m` | 実行モード (train/upload/both) | both |
| `--ip` | `-i` | M5StackのIPアドレス | - |
| `--time` | `-t` | 定期実行時刻 (HH:MM) | 03:00 |
| `--model-file` | `-f` | アップロードするモデルファイル | 最新を自動検出 |

---

## 使い方

### Webインターフェース

M5Stackが表示するIPアドレスにブラウザでアクセス:
```
http://192.168.1.XXX/
```

モデルの手動アップロード:
```
http://192.168.1.XXX/upload
```

---

## プロジェクト構造

```
.
├── MLOPS_M5STACK.ino          # M5Stack メインプログラム
├── config.h                   # 認証情報 (gitignore対象)
├── config.h.example           # 設定ファイルのテンプレート
├── model.h                    # 初期モデルデータ
├── MLops_training_tflite_uploadM5.py  # 学習・アップロードスクリプト
├── models/                    # 学習済みモデル保存ディレクトリ
│   └── YYYYMMDD/
│       ├── model_weight_YYYYMMDD.tflite
│       ├── model_weight_YYYYMMDD.keras
│       └── learning_curve_YYYYMMDD.png
└── README.md
```

---

## トラブルシューティング

### Arduino書き込みエラー

**エラー: "Sketch too big"**
- 解決: `ツール` → `Partition Scheme` → `No OTA (Large APP)`

**エラー: "A fatal error occurred: MD5 of file does not match"**
- 解決: ボードを再接続し、リセットボタンを押しながら書き込み

### WiFi接続失敗

- SSIDとパスワードが正しいか確認
- 2.4GHz WiFiを使用しているか確認 (5GHzは非対応)
- ボタンCでWiFi再接続を試行

### モデルアップロード失敗

- M5StackのIPアドレスが正しいか確認
- M5StackとPCが同じネットワークにいるか確認
- ファイアウォール設定を確認
