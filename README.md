# ProPepX Project Webpage

This folder contains a professional GitHub Pages webpage for the ProPepX manuscript.

## Files
- `index.html` — main webpage
- `style.css` — visual design and responsive layout
- `script.js` — mobile menu behavior

## How to publish on GitHub Pages
1. Copy `index.html`, `style.css`, and `script.js` into your GitHub repository root, or into a `/docs` folder.
2. Go to **Settings → Pages** in your repository.
3. Under **Build and deployment**, choose:
   - Source: **Deploy from a branch**
   - Branch: `main`
   - Folder: `/root` or `/docs`, depending on where you uploaded the files.
4. Save and wait 1–3 minutes.

## Important edits before final submission
Replace the placeholder Hugging Face model-weight link in `index.html`:

```html
<a href="#" target="_blank"><strong>Hugging Face weights</strong>
```

with your real Hugging Face repository URL.
