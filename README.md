# NovelConverter – ローカル完結オーディオブック生成システム

小説・ラノベのテキストを入力として、Qwen3-TTS系の複数TTSワーカーを使い、話者別に音声生成してオーディオブックを出力するローカルプロトタイプです。

---

## システム概要

```
[テキスト] → [前処理] → [話者分割] → [TTS生成] → [結合] → [オーディオブック(.wav/.m4b)]
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

## 使い方（E2Eフロー）

### 1. プロジェクト作成

1. http://localhost:8000 にアクセス
2. 「＋ 新規プロジェクト」をクリック
3. プロジェクト名を入力して「作成」

### 2. テキストアップロード

1. プロジェクト詳細ページで `.txt` ファイルを選択してアップロード
2. UTF-8 / UTF-8 BOM / Shift-JIS に対応

### 3. 前処理

1. 「前処理実行」ボタンをクリック
2. テキストがセグメント（段落・セリフ単位）に分割されます

### 4. 話者分割

1. 「話者分割実行」ボタンをクリック
2. ルールベース（`「」` 検出等）+ LLM（設定済みの場合）で話者を推定

### 5. セグメント確認・修正

1. 「セグメント一覧を開く」から確認画面へ
2. 信頼度が低い（赤/黄色の行）セグメントを確認
3. 「確定話者」欄を手動で修正
4. 「変更を保存」をクリック

### 6. 音声設定

1. 「音声割当を開く」から Voice Mapping 画面へ
2. 各話者にワーカー・音声パラメータを設定
3. 「すべて保存」をクリック

### 7. 音声生成

1. 「生成画面を開く」から Render 画面へ
2. 「▶ 音声生成開始」をクリック
3. 進捗バーで状態を確認（2秒ごと自動更新）
4. 完了後、「🔗 音声を結合」をクリック
5. 生成されたファイルをダウンロード

---

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
