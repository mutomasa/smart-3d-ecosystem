# Smart 3D Ecosystem — TODO

設計書: [docs/smart_3d_ecosystem_design.md](docs/smart_3d_ecosystem_design.md)

運用ルール

- 機能ごとに `feature/*` ブランチを切り、PR で `main` に取り込む。
- 開発中に見つかった問題点は GitHub Issues に起票し、`fix/*` ブランチで修正 PR を出す。
- Python は `uv`、TypeScript は `npm`。テストは `pytest` / `Vitest`。
- 関数は小さく、SOLID 原則（単一責任・依存性逆転など）に従う。

## 0. プロジェクト基盤 — `feature/project-setup`

- [x] `pyproject.toml`（Python 3.11, uv）、`.gitignore`、`.env.example`
- [x] Vite + TypeScript + Three.js の `frontend/` 雛形
- [x] `shared/schemas/policy.json`（観測・アクションの固定長スキーマ、座標系・単位）
- [x] `scripts/dev.sh`, `scripts/check_gpu.py`
- [x] GitHub Actions CI（pytest + vitest + tsc）

## Phase 1 — 3D + Boids MVP

### `feature/boids-core`（描画なしで動く数値コア）
- [x] シード付き乱数（mulberry32）
- [x] 3D 一様空間グリッド（自セル + 周辺 26 セル）
- [x] 分離・整列・結合・境界回避（滑らかな steering）
- [x] ダブルバッファの一括更新、固定 dt、max_force / max_speed、旋回率制限
- [x] テスト: 距離ゼロで NaN なし / 10,000 ステップ後も有限 / 同一シードで再現 / 個体数変更で配列整合

### `feature/renderer`（Three.js + Worker）
- [x] Web Worker で固定タイムステップ実行、Transferable ArrayBuffer のピンポン転送
- [x] InstancedMesh の魚（ローポリ自作ジオメトリ）、進行方向に姿勢
- [x] 水中シーン（ダークネイビー、フォグ、青/シアン照明）、OrbitControls
- [x] コントロール（個体数、Boids 重み、Reset/Pause、シード）、HUD（FPS / Agents / Mode / Step ms）
- [x] プリセット: Calm School / Predator Attack / Food Rush / ML vs Rules
- [x] 計測ログ（JSON ダウンロード）
- [x] 完了条件: 1,000 匹で群れ形成が目視可能

## Phase 2 — 捕食者、餌、Streamlit 連携

### `feature/predator-food`
- [x] 捕食者 1〜5 体（最寄りの群れ中心を追跡）、危険半径で逃避、捕食イベント・活性マスク
- [x] 餌 1〜10 箇所、エネルギー回復と消費、定期出現 / 手動出現
- [x] グループ分割（8〜32）とグループ要約（観測特徴量）
- [x] `boids` / `rules` モード、評価メトリクス（生存率・餌・平均エネルギー・分散・衝突）
- [x] 捕食者/餌の描画

### `feature/backend-api`
- [x] FastAPI: `/health`, `/api/config`, `/api/policies`, `/api/metrics`, `/api/experiments/reset`
- [x] `WS /ws/simulation/{experiment_id}`: policy_request / policy_response / telemetry / settings
- [x] スキーマ検証、メッセージサイズ・グループ数制限、実験 ID ごとの分離
- [x] フロントエンド dist の静的配信

### `feature/dashboard`
- [x] Streamlit: iframe で 3D 画面を埋め込み、操作パネル、メトリクス比較チャート（Plotly）
- [x] 設定同期（REST → WS push、version による last-writer-wins）

## Phase 3 — ML 方策統合 — `feature/ml-policy`

- [x] 観測正規化を Python 側 1 箇所に集約（学習・推論で共有）
- [x] PyTorch 小型 MLP（CUDA があれば GPU、なければ CPU）と一括推論
- [x] 出力後処理（方向正規化・Sigmoid 有界化）とクライアント側 0.2 秒平滑化
- [x] 模倣学習（rules の Behavior Cloning）で初期モデル、学習種別を UI に表示
- [x] フロント WebSocket クライアント: 5〜10Hz 問い合わせ、古い tick 棄却、タイムアウト/切断時 rules フォールバック、再接続
- [x] ML Latency p50/p95 計測

## Phase 4 — PPO 学習・評価 — `feature/training`

- [x] Gymnasium 環境（NumPy 版の代表グループ Boids）
- [x] SB3 PPO 学習 → PolicyNet へ重みをエクスポート、学習条件をメタデータ保存
- [x] 評価スクリプト: random / heuristic / ML、5〜10 シード、CSV/JSON レポート、学習曲線
- [x] E2E smoke test（Playwright: 描画・プリセット・バックエンド無しでの ML 劣化動作）

## Phase 5 — 演出・最適化（任意）

- [ ] 尾びれアニメーション、トレイル、ブルーム（デフォルト OFF）
- [ ] 段階的品質低下（trail → bloom → 尾びれ → 個体数）
- [x] ヘッドレス性能ベンチマーク（500 / 1,000 / 3,000 / 5,000）— `npm run bench`（Phase 1 で実施）

## 問題点（Issues で管理）

開発中に発見した問題は GitHub Issues に起票し、`fix/*` ブランチで修正した。

| Issue | 内容 | 修正 PR |
|---|---|---|
| #3 | 高密度時に近傍候補の打ち切りで分離が効かず魚群が 1 セルに崩壊 | #4 |
| #6 | 起動直後の JIT ウォームアップで adaptive dt が切り替わり再現性が崩れる | #7 |
| #9 | Rules モードの逃避で魚群が境界に張り付く | #10 |
| #13 | ML 往復レイテンシがメインスレッドの描画負荷に律速 (26ms → 2.7ms) | #15 |
| #14 | Vite proxy のバックエンドが 8000 固定 | #16 |

## PR 一覧（スタック順。上から順にマージする）

#1 基盤 → #2 Boids コア → #4 → #5 描画 → #7 → #8 捕食者・餌 → #10 → #11 Backend → #12 ML → #15 → #16 → #17 Dashboard → Phase 4 学習・評価
