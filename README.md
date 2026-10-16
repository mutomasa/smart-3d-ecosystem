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
./scripts/dev.sh   # FastAPI :8000 / Vite :5173 / Streamlit :8501
```

## テスト

```bash
uv run pytest
cd frontend && npm test
```
