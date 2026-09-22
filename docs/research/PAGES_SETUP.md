# Project website

The self-contained project page is `docs/index.html`. It does not load participant media, third-party scripts, remote fonts or analytics. The existing `web/` calibration/review application is unchanged.

The dedicated `.github/workflows/research-pages.yml` workflow publishes only `docs/index.html` and the allow-listed `docs/research/` directory. It does not publish raw captures, delivery videos or other repository files.

## GitHub Pages activation

In this repository open **Settings → Pages → Build and deployment → Source → GitHub Actions**. Then run **Actions → Research project Pages → Run workflow**, or push a change to a watched research file. This repository-administration selection cannot be performed by the file-writing GitHub connector used to prepare the project. A committed page/workflow is not proof that the site has deployed.

Expected URL after successful deployment:

https://RunyiYang.github.io/MOCAP_HAND_calibration/

The deployment job must complete successfully before this address is reported as live. Inspect the workflow run for permissions, Pages activation or environment-protection failures. Do not disable repository protection rules to bypass a failure.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. The page's research buttons link to the versioned GitHub documents, and its configuration download is a local relative link. This is a static research overview, not a running hand-pose estimator.
