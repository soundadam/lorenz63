# lorenz63

仓库说明与用法见 [README.md](README.md)。

## 本地检查

每个 commit 前跑，CI（`.github/workflows/ci.yml`）跑的是同样两条：

```bash
uv run ruff check src tests
uv run pytest
```
