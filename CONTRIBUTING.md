# Contributing

This repository extends [allenai/fluid-benchmarking](https://github.com/allenai/fluid-benchmarking). When contributing:

1. Keep changes focused; preserve compatibility with upstream APIs where possible.
2. Run `python scripts/verify_pipeline.py` before opening a PR.
3. Document new IRT types or flags in `README.md` and `docs/PROJECT.md`.

## Development install

```bash
python -m pip install -e ".[irt,ordinal]"
```
