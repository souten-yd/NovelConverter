# NovelConverter – ローカル完結オーディオブック生成システム

小説・ラノベのテキストを入力として、Qwen3-TTS系の複数TTSワーカーを使い、話者別に音声生成してオーディオブックを出力するローカルプロトタイプです。

---

## システム概要

```
[入力正規化(txt/zip/rar/epub/画像→UTF-8 text)] → [前処理] → [話者分割] → [TTS生成] → [結合] → [オーディオブック(.wav/.m4b)]
```

### 4サービス構成

| サービス | ポート | 役割 |
|---|---|---|
| Orchestrator | 8000 | Web UI + ジョブ管理 + DB |
| TTS Worker Base | 8001 | Voice Clone モード |
| TTS Worker Custom | 8002 | Custom Voice モード |
| TTS Worker Design | 8003 | Voice Design モード |

---

## 必要環境

- Python 3.10 以上
- (オプション) ffmpeg – M4B出力に必要
- (実モデル使用時) CUDA対応GPUを推奨

---

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone <repo_url>
cd NovelConverter
```

### 2. venv 作成 & 依存インストール

各サービスに独立した venv を作ります。

**Linux / macOS:**

```bash
bash scripts/setup/setup_orchestrator.sh
bash scripts/setup/setup_tts_base.sh
bash scripts/setup/setup_tts_custom.sh
bash scripts/setup/setup_tts_design.sh
```

**Windows:**

```bat
scripts\setup\setup_orchestrator.bat
scripts\setup\setup_tts_base.bat
scripts\setup\setup_tts_custom.bat
scripts\setup\setup_tts_design.bat
```

---

### 追加のシステム依存

入力正規化で以下を利用します。Pythonパッケージのインストールに加えて、必要に応じてOS側の導入を行ってください。

- OCR (`pytesseract`) を使う場合: **Tesseract OCR 本体** が必要
- RAR (`rarfile`) を使う場合: **unrar / 7zip / bsdtar** などのバックエンドコマンドが必要

未導入の場合はクラッシュではなく、アップロード結果の warning/エラー理由として表示されます（例: `RAR backend missing`）。

## 起動方法

### 全サービスを一括起動 (Linux/macOS)

```bash
bash scripts/run/run_all.sh
```

### 個別起動

**Linux/macOS:**

```bash
# 各ターミナルで実行
bash scripts/run/run_tts_base.sh
bash scripts/run/run_tts_custom.sh
bash scripts/run/run_tts_design.sh
bash scripts/run/run_orchestrator.sh
```

**Windows:**

```bat
scripts\run\run_all.bat
```

### ブラウザで開く

```
http://localhost:8000
```

---

## 使い方（UI実装ベース）

### 1. プロジェクト作成

1. `http://localhost:8000` にアクセス
2. 「＋ 新規プロジェクト」からプロジェクト名を入力して作成
3. 作成後、自動でプロジェクト詳細画面へ遷移

### 2. 原稿アップロード（非同期進捗表示）

1. プロジェクト詳細画面で OCR エンジンを選択（`tesseract` / `paddleocr` / `qwen_vl` / `ndlocr_lite`）
2. 原稿ファイルを選択して「アップロード」
3. UI は送信進捗に加えて、サーバー側取り込み進捗（SSE + ポーリングフォールバック）を表示
4. 完了後、以下のダウンロードが可能
   - 正規化テキスト (`variant=normalized`)
   - 取り込みレポートJSON (`variant=manifest`)
   - セグメント済みテキスト (`variant=segmented`, セグメント作成後)

### 3. OCR結果確認（任意）

1. 「OCR Viewer を開く」から OCR Viewer へ遷移
2. 表示モード（本文のみ / ルビ平行表示 / デバッグ）を切り替えて確認
3. 必要なら同画面から OCR パイプラインを再実行可能

### 4. 前処理

1. 「前処理実行」
2. ジョブ進捗を画面上に表示
3. 完了後、セグメント数が更新

### 5. 話者分割（通常実行 or スタジオ）

- クイック実行: プロジェクト詳細の「話者分割実行」
- 詳細実行: 「分割スタジオで開く」
  - ルール有効/無効
  - ルール順序ドラッグ
  - 高精度パイプライン / LLM主体推定 / LLMフォールバック設定
  - 「分析実行」で保存なしプレビュー
  - 「この結果を保存」で DB 反映

### 6. セグメント確認・修正

1. 「セグメント一覧を開く」
2. フィルタ（話者 / 信頼度 / 章 / 要確認 / unknown）で絞り込み
3. 各セグメントで以下を編集
   - 確定話者（候補選択または直接入力）
   - セグメント種別
   - 修正理由（任意）
4. 「変更を保存」で一括反映（整合性再チェックが非同期実行）

### 7. 音声設定（2画面）

- **Voice Design Studio（推奨）**
  - 話者ごとのタブ式編集
  - AI音声提案（LLM利用時）
  - 参照音声アップロード（clone向け）
  - 音声プレビュー生成
  - 話者単位で保存
- **シンプル設定（voice_mapping）**
  - 一覧フォームで全話者の設定をまとめて編集
  - 「🔊 テスト生成」で個別確認
  - 「すべて保存」で一括保存

### 8. 音声生成・再試行・結合

1. Render 画面で「▶ 音声生成開始」
2. 2秒ごとに進捗（完了/失敗/未処理）を更新
3. 失敗がある場合は「🔁 失敗を再試行」で failed セグメントを pending に戻す
4. 「🔗 音声を結合」で chapter WAV / full WAV（可能なら M4B）を生成
5. 生成ファイル一覧からダウンロード

### READMEとの差分（今回反映）

従来の README は最短E2E中心でしたが、実UIには以下の機能があり、上記手順へ追記しました。

- 話者分割スタジオ（プレビュー保存フロー）
- OCR Viewer（表示モード切替とパイプライン再実行）
- Voice Design Studio（AI提案・プレビュー・参照音声アップロード）
- Render 画面の「失敗再試行」導線
- アップロード進捗の非同期ジョブ監視（SSE/ポーリング）

## 実モデル（Qwen3-TTS）の使い方

### 依存パッケージの追加インストール

各ワーカーの `requirements.txt` のコメントアウトを解除して再セットアップ：

```
torch>=2.1.0
transformers>=4.40.0
soundfile>=0.12.1
numpy>=1.24.0
```

### 環境変数で有効化

```bash
# TTS Base (voice clone)
export TTS_BASE_USE_REAL=true
export TTS_BASE_MODEL_PATH=Qwen/Qwen3-TTS  # or local path

# TTS Custom (custom voice)
export TTS_CUSTOM_USE_REAL=true
export TTS_CUSTOM_MODEL_PATH=Qwen/Qwen3-TTS

# TTS Design (voice design)
export TTS_DESIGN_USE_REAL=true
export TTS_DESIGN_MODEL_PATH=Qwen/Qwen3-TTS
```

---

## LLM連携（話者分割精度向上）

OpenAI互換APIサーバーが利用可能な場合、以下を設定すると話者推定精度が上がります：

```bash
export LLM_API_URL=http://localhost:11434/v1   # Ollama例
export LLM_MODEL=llama3.2                       # モデル名
export LLM_API_KEY=                             # API Key（不要な場合は空）
```

未設定の場合はルールベースのみで動作します。

### llama-server が見つからない場合

`llama-server binary not found at 'llama-server'` が出る場合は、`llama.cpp` の `llama-server` 実行ファイルを配置し、環境変数を設定してください。

```bash
export LLAMA_SERVER_BIN=/path/to/llama-server
```

Docker Publish (GitHub Actions) のビルド時に `ai-dock/llama.cpp-cuda` の最新 CUDA 12.1 アーティファクトを取得して `/opt/llama-cpp/bin/llama-server` を同梱します。

さらに RunPod 起動後に万一 `llama-server` が見つからない場合、エントリポイントが起動ログで警告を表示し、`/opt/llama-cpp/bin/llama-server` へのランタイムフォールバックダウンロードを自動で試行します（失敗時はルールベースのみ継続）。

---

## ディレクトリ構成

```
NovelConverter/
  app/
    orchestrator/          # メインサービス (port 8000)
      routers/             # FastAPI ルーター
      services/            # ビジネスロジック
      templates/           # Jinja2 HTML テンプレート
      static/              # CSS / JS
    workers/
      tts_base/            # Voice Clone ワーカー (port 8001)
      tts_custom/          # Custom Voice ワーカー (port 8002)
      tts_design/          # Voice Design ワーカー (port 8003)
    shared/                # 共有モジュール (models, schemas, logger, db)
  data/
    projects/              # アップロードテキスト
    outputs/               # 生成音声ファイル
    temp/                  # 一時ファイル
    references/            # 参照音声 (voice clone用)
    novelconverter.db      # SQLite DB
  scripts/
    setup/                 # venv作成スクリプト
    run/                   # 起動スクリプト
  samples/                 # サンプルファイル
  docs/                    # ドキュメント
```

---

## 環境変数一覧

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/novelconverter.db` | DB接続URL |
| `ORCHESTRATOR_PORT` | `8000` | Orchestratorポート |
| `TTS_BASE_URL` | `http://localhost:8001` | Base WorkerのURL |
| `TTS_CUSTOM_URL` | `http://localhost:8002` | Custom WorkerのURL |
| `TTS_DESIGN_URL` | `http://localhost:8003` | Design WorkerのURL |
| `TTS_BASE_PORT` | `8001` | Base Workerポート |
| `TTS_CUSTOM_PORT` | `8002` | Custom Workerポート |
| `TTS_DESIGN_PORT` | `8003` | Design Workerポート |
| `TTS_BASE_USE_REAL` | `false` | 実モデル使用フラグ |
| `TTS_CUSTOM_USE_REAL` | `false` | 実モデル使用フラグ |
| `TTS_DESIGN_USE_REAL` | `false` | 実モデル使用フラグ |
| `TTS_BASE_MODEL_PATH` | `Qwen/Qwen3-TTS` | モデルパス |
| `TTS_CUSTOM_MODEL_PATH` | `Qwen/Qwen3-TTS` | モデルパス |
| `TTS_DESIGN_MODEL_PATH` | `Qwen/Qwen3-TTS` | モデルパス |
| `LLM_API_URL` | `` | LLM APIのURL（空=ルールベースのみ）|
| `LLM_API_KEY` | `` | LLM APIキー |
| `LLM_MODEL` | `gpt-4o-mini` | LLMモデル名 |

---

## REST API リファレンス

### Orchestrator (port 8000)

| Method | Path | 説明 |
|---|---|---|
| POST | `/api/projects` | プロジェクト作成 |
| GET | `/api/projects` | プロジェクト一覧 |
| GET | `/api/projects/{id}` | プロジェクト詳細 |
| POST | `/api/projects/{id}/upload_text` | テキストアップロード |
| POST | `/api/projects/{id}/preprocess` | 前処理実行 |
| POST | `/api/projects/{id}/segment_speakers` | 話者分割実行 |
| GET | `/api/projects/{id}/segments` | セグメント一覧 |
| POST | `/api/projects/{id}/segments/update` | セグメント一括更新 |
| GET | `/api/projects/{id}/speakers` | 話者一覧 |
| POST | `/api/projects/{id}/voice_mappings` | 音声設定保存 |
| POST | `/api/projects/{id}/render` | 音声生成開始 |
| GET | `/api/projects/{id}/render/status` | 生成状態取得 |
| POST | `/api/projects/{id}/render/retry_failed` | 失敗セグメントリセット |
| POST | `/api/projects/{id}/render/speaker/{name}` | 話者別再生成 |
| POST | `/api/projects/{id}/merge` | 音声結合 |
| GET | `/api/projects/{id}/artifacts` | 生成ファイル一覧 |

### TTS Workers (port 8001/8002/8003)

| Method | Path | 説明 |
|---|---|---|
| GET | `/health` | ヘルスチェック |
| GET | `/models` | 利用可能モデル一覧 |
| POST | `/warmup` | モデルウォームアップ |
| POST | `/synthesize` | 音声合成 |

---

## 既知の制約

[docs/constraints.md](docs/constraints.md) を参照してください。

---

## ライセンス

MIT License

## Windows 11 ローカル起動（AMD GPU / 非 ROCm）

本リポジトリでは `run_local_windows.bat` を追加し、`.venv` の自動構築・再利用で起動できます。

### 使い方

1. Python 3.11 をインストール（`py -3.11` が使える状態）
2. プロジェクトルートで `run_local_windows.bat` を実行

初回は以下を自動実行します。
- `.venv` 作成
- `requirements-local-windows.txt` のインストール
- モデル保存ディレクトリ作成
- Qwen3-TTS / Gemma GGUF / NDLOCR モデルの不足分ダウンロード
- バックエンド判定（LLM Vulkan優先、PyTorch/ONNX DirectML優先、失敗時CPU）

2回目以降は `.venv` と既存モデルを再利用し、不足分のみ補修します。
