# CLAUDE.md

## Project Overview

Academic personal website for Shi Gu (Associate Professor, Zhejiang University, Brain and Intelligence Lab). Hosted on GitHub Pages at https://guslab.org/.

## Tech Stack

- **Pure static site**: vanilla HTML5, CSS3, JavaScript — zero dependencies, no build tools
- **Hosting**: GitHub Pages (deploy by pushing to `main`)

## Project Structure

```
├── index.html            # Home/About page (bio, news feed)
├── research.html         # Research goals with tabbed interface
├── team.html             # Team members and alumni
├── publications.html     # Publications by year
├── shares.html           # Blog posts and talk recordings (filterable grid)
├── style.css             # Single global stylesheet
├── script.js             # Shared JS (tab switching, filtering)
├── assets/               # Images and icons (avatar, backgrounds, goal figures)
└── shares/               # Content subdirectories for blog/talk pages
    └── <topic>/          # Each has content.html + featured.png
```

## Conventions

- All pages share the same `<header>`/`<nav>`/`<footer>` structure — copy it consistently when adding pages
- Subpages use `light-header` class; index uses the default dark header with banner
- CSS uses flat class naming (`.tab-button`, `.card`, `.card-tag`), not BEM or CSS modules
- JS is minimal: tab switching via `showTab(tabId)`, filtering via `data-filter`/`data-type` attributes
- Share subpages reference root styles with `../../style.css` (two levels deep)
- Responsive layout via CSS flexbox/grid; cards use `minmax(320px, 1fr)`
- Color palette: dark header `#1a1a1a`, blue accent `#007BFF`, yellow hover `#FFC107`

## Working with the Site

- No install or build step — open HTML files directly or use any local server
- Changes deploy automatically when pushed to `main` via GitHub Pages
- Test responsiveness across breakpoints when modifying layout
