# 文脈付き音声台本フロー（opt-in CLI）

## 目的と実装範囲

音声方式を決める前に、原文・人物ID・発話種別・演技指示を分離して検証する。
フロンティアの OpenAI-compatible chat/completions endpoint で構築・評価し、後で同じ
JSON契約を使うローカルサーバーへ切り替える。標準ライブラリのみで動作する。
既存WebUI・DB・話者分割・TTSワーカーは変更しない。このCLIの結果は旧renderへ
自動投入しない。現段階の完成物は**検証済み台本とTTSジョブ計画**であり、音声ではない。

## 実行

```bash
python scripts/audiobook_script.py prepare /path/book.epub /path/book-work
# 利用するプロバイダのOpenAI互換URL、利用可能なフロンティアモデルを明示する。
export SCRIPT_LLM_API_URL=https://YOUR-PROVIDER/v1
export SCRIPT_LLM_MODEL=YOUR-FRONTIER-MODEL
export SCRIPT_LLM_API_KEY=YOUR-KEY
python scripts/audiobook_script.py analyze /path/book-work
```

`analyze`は本文を指定したAPIへ送信する。prepare/planは通信しない。キーは保存しない。
接続先は既存LLM管理と独立しているため、ローカルモデルのロードに上書きされない。
プロバイダによっては独自APIや追加パラメータが必要で、その場合はcall_modelの
アダプタを追加する。JSON出力に失敗した場合は停止する（ルール推定で成功扱いにしない）。

APIが使えない場合、requests.jsonのsystemと各batchに既存characters台帳を添え、
フロンティアモデルでJSONを作り `responses/batch-00000.json` 等へ保存する。
後続batchにはそれまでの台帳を渡す。応答は次のenvelope形式で保存する。request_digestはrequests.jsonの該当batchからコピーする。全応答が揃ったら取り込む。

```json
{"request_digest":"コピーした値", "response":{"characters":[], "annotations":[]}}
```

prepare時のbatch/context設定をanalyzeが引き継ぐ。設定変更時はprepareからやり直す。別の本文・サンプル・プロンプトの応答は拒否する。

```bash
python scripts/audiobook_script.py analyze /path/book-work --responses /path/responses
```

これは応答の取り込みであり自動モデル実行ではない。応答を修正して再取り込みした場合、
古いキャッシュを使わない。API呼び出しは検証済みbatchだけを再利用する。
API失敗時は同じコマンドで再開できる。失敗した再解析で古いscript/planを完成物として残さない。プロンプト・本文・台帳・モデル・URLが変われば
キャッシュが変わる。出力ディレクトリは作品ごとに分け、同時実行しない。

## 成果物と確認

- source.json: spine順の段落、href/element_id、原文、ルビ（本文に読みを二重挿入しない）、原本SHA256。
- requests.json: 全文を省略しない対象blockと前後8blockの文脈。標準上限6000文字。
- api/script.json または imported/script.json、ルートのscript.json: 人物台帳と全区間。
- api/review.json または imported/review.json: 不明・低確信・根拠不足等の確認一覧。
- tts_plan.json: 検証済み生成ジョブ。未解決が一件でもあればready=false、jobsは空。

start/endは抽出段落内のPython Unicode文字位置（end排他的）。原EPUBバイト位置ではない。
全段落について連続・欠落なし・重複なしを検証。モデルは本文を書き換えない。
抽出対象はXHTML本文の段落等。画像のみページはOCRせずskippedへ明記する。
非linear/navは除外する。表紙・目次・奥付など段落として残った部分はモデルで
heading/non_speechと判定し、レビューで確認する。特殊なEPUBの表・裸テキスト混在等は
抽出結果との照合が必要。XMLとして不正なHTMLは推測で修復せずエラーにする。

人物名・別名は安定IDに結び付ける。既存IDの別人物への書き換えを拒否し、異なるIDに
同一名称/aliasがある場合はplanを止める。意味的な別名解決の正しさを自動保証するもの
ではない。同一人物の別IDが見つかったら全セグメントと台帳を一緒に修正する。

地の文はnarrator、心内語は本人のIDにする。「と思った」という描写自体を心内語扱い
しない。呼びかけは聞き手として扱い、短い相槌や複数人の同時発話は断定しない。
感情emotionと話し方deliveryは人物IDと独立する。根拠引用の実在は検証できるが、
根拠の解釈が正しいかは別途評価が必要。confidenceはモデルの自己申告である。

## レビューと声の固定

script.jsonで話者/感情/種別を修正する。原文、範囲、並び順は変えない。
unknownは具体的な人物へ解決する。needs_reviewを手動で消してゲートを回避せず、
確認したsegment全体のdigestをapprovals.jsonに登録する。

```python
from app.orchestrator.services.audiobook_script import digest, save
# s は実際に確認済みのsegment。全件を機械的に承認しない。
approvals[s['id']] = digest(s)
save('approvals.json', approvals)
```

voices.jsonは人物ID→固定音声設定。例（未決定の値を実際の選定結果へ置き換える）:

```json
{
  "narrator": {"provider": "chosen-engine", "voice_id": "narrator-01", "revision": "v1"},
  "c01": {"provider": "chosen-engine", "voice_id": "character-01", "revision": "v1"}
}
```

```bash
python scripts/audiobook_script.py plan /path/book-work voices.json --approvals approvals.json
```

同一人物の会話/心内語/感情変化でvoiceを変えない。ジョブにはkindも保持する。感情/話し方の確信度は人物推定とは別に検証し、未指定の非neutral演技はレビュー対象とする。TTS上限に合わせて
`--max-chars 200`で分割してもID・感情・原文位置は維持する。句読点付近を優先し、ルビの親文字を途中で切らない。親文字自体が上限より長い場合はエラーにする。トークン上限は最終エンジンのアダプタで対応する。
参照音声、モデル、発音辞書、seed等をvoice設定へ含めるとキャッシュキーに反映される。
実ファイルの差し替えもrevision更新が必須。ルビはsource.jsonに親文字のstart/end付きで保持し、各TTSジョブにもジョブ内位置と読みを引き継ぐ。TTSエンジン選定後に発音入力へ適用する（現時点では原文textを書き換えない）。

## 音声方式決定後の接続契約

ready=trueのjobsを順番に読み、voiceのprovider/voice_id/revisionを固定して合成する。
textを省略せず、emotion/deliveryは演技指定として渡す。未対応の演技は能力情報で
明示する。同じcache_keyの音声だけ再利用し、失敗区間だけ再生成する。
出力音声はサンプルレート/チャンネル/音量を揃え、欠落・重複・順番・無音を検証して
連結する。このTTSアダプタと聴取/ASR評価はまだ実装していない。台本のID固定だけで
実際の声質の同一性や読み間違いのなさを保証するものではない。

## 添付EPUBでの確認と評価方法

提供された縦書き日本語EPUBで、spine順抽出、inline要素の連結、rb/rt分離、
画像のみページの除外を実行確認した。本文/EPUB/抽出台本はリポジトリへ含めない。
冒頭には呼称（姉様等）、双子の会話、発話者を断定しづらい短い相槌、引用された
用語が混在する。これを区別するプロンプトと未確定ゲートを設けた。
外部API未設定の環境では全巻のモデル推定精度・音声品質は未検証。

ローカルへの移行前に、呼称、相槌、心内語、引用、視点切替、章またぎ、同時発話を
含む人手確認済み区間を評価セットにする。フロンティア出力も正解扱いせず、
話者ID一致率、kind一致率、未確定率、別名の誤統合、感情の過剰推定、原文被覆率、
費用/時間を別々に比較する。必要に応じてcontextやbatchを調整する。

```bash
python -m unittest discover -s tests -p test_audiobook_script.py -v
```


## サンプル評価と曖昧な発話の扱い（2026-10-02）

```bash
python scripts/audiobook_script.py prepare /path/book.epub /path/sample-work \
  --sample-range 23:67 --sample-range 314:337 --sample-range 3046:3064
```

範囲は抽出後のblock配列に対する0始まり/end排他。作品ごとに異なる。
サンプルの離れた範囲同士は文脈として接続しない。同一範囲内のspine切替前後は
文脈参照を維持する。抽出内容/範囲のdigestを台本に結び付ける。

モデルの位置計算ミスを避けるため、段落全体は`whole_block:true`で返せる。
複数区間はstart/endまたは原文完全一致のtextを順番に返す。text方式も重複/欠落/改変を拒否する。

「え」「あ」などで人物が不明な場合、標準は従来どおり停止する。
演出判断として代読を選ぶ場合に限り、`--unresolved-policy narrator --decisions decisions.json`
を指定する。decisionsは以下の形式。通常のapprovalsだけでは不明話者を解除できない。

```json
{
  "segment-id": {
    "segment_digest": "確認したsegment全体のdigest",
    "action": "narrator_fallback",
    "reason": "個別話者の根拠が不足しているため、代読という演出を選択"
  }
}
```

推定speaker_idはunknownのまま、render_speaker_idだけnarratorにする。
候補人物candidatesと根拠reason/evidenceを保持する。修正後はdigestが変わり
古い代読指定が無効になる。kindまでunknownの場合は代読指定で通過させない。

ダッシュのみの沈黙をdialogueにするとpossible_silenceのレビュー理由を付ける。
見出し/沈黙を除外する場合もexcluded_segmentsに原文と理由を残す。
現時点の沈黙は音声ジョブを生成しないため、秒数付きの間は最終TTS編集段階で決める。

今回のサンプル評価と未評価範囲は[評価記録](audiobook_sample_evaluation.md)を参照。
