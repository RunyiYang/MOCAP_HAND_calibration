# GT Calib review site

## Final nine-video page

The final handoff page is `/final-nine/`. `npm run build` recreates
`public/downloads/final-nine/` from the tracked
`../final_9_video_delivery/`; the generated web mirror stays ignored so Git has
only one authoritative copy. Then serve the completed tree with byte-range
support:

```bash
cd web
npm run build
cd ..
uv run gt-calib-delivery serve --host 127.0.0.1 --port 8811
```

The page reads `/downloads/final-nine/manifest.json` and requires exactly nine
matching video IDs before showing the manifest as verified. Its four paired
groups offer synchronized play/reset controls; the ninth player is no-glove
world-calibration evidence.

The mirror step is fail-closed. Before replacing an existing published tree it
stages a non-symlink copy and verifies all of the following:

- manifest schema/status, exactly nine unique video IDs/files and orders 1–9;
- every manifest byte count and SHA-256, including poster/metrics/frame-map,
  camera calibration and optional calibration-workbench assets;
- `validation.json` is a clean nine-video pass;
- `SHA256SUMS.txt` covers every regular artifact exactly once;
- every asset is within the Cloudflare 25 MiB per-file limit.

Only after the staged copy passes does the builder swap it into
`public/downloads/final-nine/`, retaining the previous complete tree for
rollback during the swap. A failed validation leaves the previous publication
unchanged. The final-page HTML/CSS/JS and all mirrored delivery files are also
included in `data/asset-manifest.json`.

## Local review with video seeking

Build the static assets, start the read-only byte-range server, then open the BVH page:

```bash
cd /home/runyi/Project/hands_reloc/GT_calib
python web/build_site.py
python local_review_server.py
```

```text
http://127.0.0.1:8811/bvh/?take=02
```

The dedicated local server binds to localhost by default and supports the HTTP
Range responses required for reliable MP4 timeline seeking.

This directory contains the Cloudflare Workers Static Assets delivery site for
exposure-time MOCAP/video matching plus the cross-take glove/MOCAP fusion and
diagnostic comparison.

## Build

```bash
git lfs pull
npm ci
npm run build
npm run dev
```

The default build is a standard-library-only deployment assembly. It validates
the checked-in `public/` tree against `data/asset-manifest.json`, validates and
mirrors the tracked final delivery, then writes the aggregate asset manifest
atomically. It does not read raw acquisition data or ignored `outputs/`, so it
is the command for clean clones and Cloudflare Git builds. Run `git lfs pull`
first: the final nine and the checked-in legacy review MP4s are Git LFS objects,
and unresolved pointer files intentionally fail byte/hash validation instead
of producing a broken website.

To regenerate the older dataset-review assets from the local raw/derived
workspace, use the explicit source build:

```bash
npm run build:source
```

`build:source` reads MOCAP/video alignment, MOCAP-root fusion,
wrist-preserved diagnostic metrics, the held-out wrist-semantics audit and the
request CSV. It requires the project Python dependencies plus the ignored raw
and `outputs/` directories, copies only allow-listed assets, and then performs
the same final-delivery mirror and Cloudflare size checks. It is not the
Cloudflare clean-clone build command.

For a manual UV-only refresh, `uv run gt-calib-delivery publish-web` remains
available, but it is not required before `npm run build` and it refuses to
overwrite an existing mirror.

## Deploy

```bash
npx wrangler login
npx wrangler whoami
npm run deploy:dry
npm run deploy
```

The review videos contain an identifiable participant and lab scene. Confirm
the intended audience before deployment; use Cloudflare Access when the site is
not meant to be public.

The checked-in configuration sets both `workers_dev` and `preview_urls` to
`false`. This allows the first authenticated upload to create/update the Worker
without publishing a public `workers.dev` or preview URL. After that upload:

1. In **Workers & Pages → gt-calib-dataset-review → Access**, protect the whole
   Worker and include only the reviewer emails/groups.
2. Add a Custom Domain in **Settings → Domains & Routes**. This repository leaves
   routes out of Wrangler so that the domain can be managed in the dashboard.
3. Test an unauthenticated request in a private browser window; it must reach an
   Access login/block page before any HTML or video bytes are returned.
4. Only then copy the protected URL into Feishu.

Do not use `wrangler deploy --temporary` for this dataset: temporary preview
accounts are not the intended authenticated delivery path.

## Content contract

- Take 01 / Take_000 is calibration/self-fit.
- Take 02 and Take 03 are frozen-profile holdouts.
- The default view keeps the original Skeleton_0/1 21 joints per hand and adds
  a display-only rear wrist module proxy. In the formal videos it is a
  side-coloured dot with a white centre; in the held-out audit contacts it is an orange
  diamond. It is not a measured joint or GT.
- The unmodified raw 21-joint H264/contact remains downloadable beside every
  recommended proxy video, and both views must have byte-identical alignment
  CSVs.
- The BVH manifest must be exported from the same rear-proxy alignment directory
  as the published default H264 so its recorded video SHA256 resolves exactly.
- Publication is fail-closed unless every source BAG color message and source
  MP4 frame passes the decoded-content index gate (`--frame-content-samples 0`);
  the 31-frame mode is only a fast regression check.
- The fusion view is MOCAP-wrist-SE(3)-conditioned glove articulation; the
  wrist-preserved view is a diagnostic baseline.
- The 10–17 ms visual-motion residual is QA-only and must never be written back
  as a per-take clock adjustment.
- Target-take finger joints do not enter the frozen canonical R/scale fit, but
  target-take MOCAP wrist position and orientation do enter every fused frame.
- The page must not describe fusion metrics as independent glove wrist or full
  6DoF accuracy, and must show the high P95 alongside the improved median.
- The page must distinguish frame-complete timestamp interpolation from
  hardware-level zero-time-error or independent 2D reprojection accuracy.
- Files under `outputs/feishu_dataset_understanding/` that contain the older
  per-take-fit metrics are not copied into the site.
