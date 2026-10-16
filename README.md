# Smart 3D Ecosystem — Boids + Machine Learning

数千体の魚群が 3D 空間で群れを形成し、捕食者を回避しながら餌を探索するエコシステムの可視化デモ。
ルールベースと学習済み方策の差を定量評価する。

- 設計書: [docs/smart_3d_ecosystem_design.md](docs/smart_3d_ecosystem_design.md)
- TODO: [todo.md](todo.md)

## セットアップ

```bash
uv sync --all-extras          # Python 3.11 + 依存（training extra 含む）
cd frontend && npm install    # Three.js / Vite / Vitest
uv run python scripts/check_gpu.py
```

## 開発サーバー

```bash
./scripts/dev.sh                    # FastAPI :8000 / Vite :5173 / Streamlit :8501
BACKEND_PORT=8010 ./scripts/dev.sh  # 8000 が使用中の場合（Vite proxy / Streamlit も追従）
```

Vite を単独で動かす場合は `BACKEND_URL=http://localhost:8010 npm run dev` でプロキシ先を変更できます。

## テスト

```bash
uv run pytest
cd frontend && npm test
```

## ダッシュボード (Streamlit)

```bash
API_URL=http://localhost:8000 SCENE_URL=http://localhost:5173/ uv run streamlit run dashboard/app.py
```

- 3D シーン（FastAPI が配信する Vite ビルド、または Vite dev server）を iframe で埋め込み、`?experiment=<ID>` で実験 ID を共有
- サイドバーから設定をシーンへ push（REST → WebSocket、version 付き last-writer-wins）
- 1 秒ごとのテレメトリから KPI・時系列・モード別（Boids / Rules / ML）の run 最終値の平均±標準偏差を表示

## 学習・評価 (Phase 3 / 4)

```bash
# 1) 模倣学習: rules を Behavior Cloning して最初の ML 方策 (bc-v1) を作る（A1000 で約 45 秒）
uv run python -m training.imitation                       # -> backend/models/policy.pt (+ .json)

# 2) PPO: Gymnasium 環境 (training/env.py) で bc-v1 から初期化して学習
uv run python -m training.train_ppo --timesteps 500000 --init-from backend/models/policy.pt \
    --version ppo-v1 --out backend/models/ppo.pt          # 学習曲線: outputs/ppo/ppo-v1/learning_curve.csv

# 3) 評価: 学習と分離したシード (10000〜) で random / rules / 各チェックポイントを比較
uv run python -m training.evaluate --seeds 10 \
    --checkpoint bc-v1=backend/models/policy.pt --checkpoint ppo-v1=backend/models/ppo.pt
#    -> outputs/eval/<timestamp>/{results.csv,summary.csv,summary.json}, outputs/eval/latest.json

# 4) サーバーで使うモデルを切り替える
POLICY_CHECKPOINT=backend/models/ppo.pt uv run uvicorn backend.main:app --port 8000
```

- 観測の正規化 (`backend/normalization.py`) と出力の後処理 (`backend/actions.py`) は学習・推論で同一コード
- チェックポイントが無い / 壊れている場合、`ml` は **`Heuristic policy (untrained)`** として rules を返す（学習済みと偽らない）
- 学習環境は代表 1 グループ（既定 48 匹）の NumPy 実装。ブラウザのシミュレーションとは規模・一部実装が異なる（sim-to-sim ギャップ）ため、最終比較はブラウザ側のテレメトリでも確認する
- `backend/models/` と `outputs/` は Git 管理外
