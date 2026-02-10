---
model: sonnet
---

Analyze all modified and untracked files, group them logically, and create one or more conventional commits.

### Conventional Commits

**Format:** `<type>(<scope>): <description>`

Title must be **< 72 characters**.

#### Types

- feat: new feature
- fix: bug fix
- refactor: code change that neither fixes a bug nor adds a feature
- style: formatting, missing semicolons, etc.
- perf: performance improvement
- docs: documentation
- chore: maintenance tasks
- ci: continuous integration changes
- test: adding or correcting tests

#### Scopes

Use one of these fixed scopes. Omit the scope only when a change spans too many areas to pick one.

| Scope | Covers |
|---|---|
| `gallery` | Masonry grid, Gallery.astro, PaintingCard.astro |
| `lightbox` | PhotoSwipe, Lightbox.astro, lightbox.js |
| `filters` | FilterPanel.astro, filters.js, chip logic |
| `artist` | "L'artista" about page (artista.astro) |
| `layout` | Base.astro, HTML structure, meta tags |
| `nav` | Header.astro, navigation |
| `styles` | global.css, theme, typography, variables |
| `data` | paintings.yaml, data schema changes |
| `images` | Image preprocessing, CDN config, upload scripts |
| `scripts` | Build/utility scripts (scripts/ directory) |
| `ci` | GitHub Actions, deploy.yml |
| `config` | astro.config.mjs, package.json, tooling |
| `spec` | PRD.md, project documentation |

#### Examples

```
feat(gallery): add proportional sizing based on real cm
fix(lightbox): restore scroll position on close
style(styles): adjust dark theme contrast values
chore(data): add 20 new paintings to YAML
perf(gallery): defer offscreen images with IO
ci: add GitHub Pages deploy workflow
docs(spec): clarify filter logic in PRD
```

### Workflow

1. Run `git status` to see overall repository state. If there are no changes (staged or unstaged), exit.
2. Run `git diff` and `git diff --stat` to analyze all unstaged changes.
3. Run `git diff --staged` and `git diff --stat --staged` to analyze already staged changes.
4. Run `git log --oneline -10` to review recent commit patterns.
5. Group the changed files logically by scope/purpose. If all changes belong to the same logical unit, make a single commit. If changes span multiple unrelated scopes, split them into separate commits (e.g., a style change and a new gallery feature should be two commits).
6. For each logical group, in order:
   a. Stage only the files for that group with `git add <file1> <file2> ...`
   b. Write a concise commit message (72 chars max for first line). Include a body if the changes are complex.
   c. Create the commit.
7. After all commits, run `git log --oneline -5` to confirm the result.
