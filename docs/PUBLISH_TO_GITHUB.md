# Publish this project to your own GitHub account

## 1. Create an empty repository on GitHub

1. Go to [github.com/new](https://github.com/new).
2. Choose a name (e.g. `adaptive-benchmarking-ordinal` or `fluid-benchmarking-ordinal`).
3. **Do not** add a README, `.gitignore`, or license (this repo already has them).
4. Create the repository and copy the HTTPS or SSH URL, e.g. `https://github.com/aidenzhou/adaptive-benchmarking-ordinal.git`.

## 2. Point your local clone at the new remote

From your `fb-ordinal` directory (the one with this README):

```bash
# Optional: rename remote if you still use allenai as "origin"
git remote rename origin upstream

# Add your repo
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push main (or your default branch)
git branch -M main
git push -u origin main
```

If you prefer to keep `origin` as Allen AI and only push your fork:

```bash
git remote add myrepo https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u myrepo main
```

## 3. Update links in the README

Search and replace:

- `YOUR_USERNAME` / `YOUR_REPO` in `README.md`
- Same in `docs/RESUME_BLURB.md` if you use it

## 4. Optional: GitHub repository settings

- Add a short **description** on GitHub: e.g. *Ordinal & continuous IRT extension of Fluid Benchmarking for adaptive LLM evaluation.*
- Add **topics**: `llm`, `benchmark`, `irt`, `adaptive-testing`, `pytorch`, `evaluation`.
