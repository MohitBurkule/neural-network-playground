# Deploying the Neural Network Playground

The whole project is a static site. `npm run build-all` compiles the main app
**and** every lab into a self-contained `dist/` folder, which is what gets
served. Node 22 is pinned in `.node-version`.

## Option A — Cloudflare Pages (Git-connected, recommended)

Auto-deploys on every push / PR.

1. Cloudflare dashboard → **Workers & Pages → Create → Pages → Connect to Git**.
2. Pick the `mohitburkule/neural-network-playground` repo.
3. Build settings:
   - **Framework preset**: None
   - **Build command**: `npm run build-all`
   - **Build output directory**: `dist`
   - **Environment variable**: `NODE_VERSION = 22`
4. Save and Deploy. Production branch can be `main`; PRs get preview URLs.

## Option B — Wrangler direct upload (no build on CF)

```bash
npm install -g wrangler
wrangler login
npm run build-all
wrangler pages deploy dist --project-name=neural-network-playground
```

## Option C — GitHub Pages

```bash
npm run build-all
# publish the dist/ folder to the gh-pages branch (e.g. via peaceiris/actions-gh-pages
# or `git subtree push --prefix dist origin gh-pages`)
```

## Routes

The site is a flat set of files in `dist/`:

- `/` (or `/index.html`) — the main unified playground
- Labs: `/cnn.html`, `/transformer.html`, `/autoencoder.html`, `/rnn.html`,
  `/gan.html`, `/clustering.html`, `/rl.html`, `/dtree.html`, `/dimred.html`,
  `/svm.html`, `/glm.html`, `/gp.html` (plus diffusion/word2vec when built)

No SPA rewrites or redirects are needed.

## Notes specific to this app

- **three.js** (used by the in-page 3D mode and is referenced by the 3D
  visualization) is loaded from a CDN at runtime, so visitors need network
  access to that CDN. Everything else is bundled and served locally.
- `analytics.js` calls Google Analytics; set your own GA ID or remove it before
  a public launch.
- CI (`.github/workflows/ci.yml`) runs `npm run build-all` + `npm test` on every
  push so a broken build is caught before deploy.
