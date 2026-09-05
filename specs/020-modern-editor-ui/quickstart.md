# Quickstart: Modern Editor UI

Validate the restyle against `data-model.md` and the contracts in `contracts/`. Do not treat Standalone as a product.

## Prerequisites

- macOS (primary), CMake build with the existing OpenYourBox targets
- Ability to launch **OpenYourBox Standalone**
- `.ignore/` is gitignored (stills go in `.ignore/visual-refs/`)

Configure and build (from the repo root; adjust `-B` if you already have a build tree):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target OpenYourBox_Standalone -j
cmake --build build --target OpenYourBoxVisualLanguageTests -j
ctest --test-dir build --output-on-failure -R VisualLanguage
```

Standalone binary (typical): `build/OpenYourBox_artefacts/Release/Standalone/OpenYourBox.app`


## 1. Before still

1. Build and launch Standalone **before** restyle (or from `main` if restyle is not yet merged).
2. Size the window like a typical plugin (~900×600 class).
3. Capture **window-only** → `.ignore/visual-refs/before-plugin.png`
4. Repeat at a large size → `.ignore/visual-refs/before-large.png`

## 2. Token and font smoke (CTest)

```bash
ctest --output-on-failure -R VisualLanguage
```

Expect: Live, Frozen, accent, and danger pairwise distinct; two Inter weights registered in the test table (or equivalent assertions in `Tests/VisualLanguageTests.cpp`).

## 3. Restyled idle

1. Build Standalone with the restyle.
2. At plugin size: graph is the majority of the window; page is dark; library overlay and right inspector are reachable.
3. Confirm Inter (not the toolkit default face) and outline icons (not emoji, not Codicons).
4. Window-only still → `.ignore/visual-refs/idle-plugin.png` and `idle-large.png`

## 4. First-session path

1. Place a processing box; wire Audio In → box → Audio Out.
2. Click the box: Parameters opens (one click).
3. Move Dry/Wet: instrument slider, not a stock thick bar.
4. Click count for this path matches today except optional extra section clicks that are obviously better — Parameters-on-select stays one click.

## 5. Live vs Frozen

1. Have a learned Live box and a Frozen box on the canvas.
2. Without reading labels, tell them apart in under one second (cool vs warm + lock).
3. Selection/focus accent must not look like the Live fill.
4. Still → `.ignore/visual-refs/mixed-live-frozen.png`

## 6. Inspector surfaces

Open Info (analysis well), Parameters, training Library, Capture, Train, Presets. Each uses instrument tabs/trees/buttons. Still per tab as in the screenshot contract.

## 7. Illegal cable

Attempt an illegal connection: cable is unambiguously red; tooltip still explains the mismatch.

## 8. Not a toolkit, not a code editor

Side-by-side `before-*` vs `idle-*`: reviewers pick before as “generic toolkit.”  
Side-by-side idle vs a VS Code still: shared **materials** (dark, quiet separators) are OK; matching activity bar / terminal / file tabs is a **failure**. Nobody should say “VS Code with a graph.”

## 9. Plugin vs Standalone

Open the same build as AU/VST3 in a DAW. Materials match Standalone. Density may scale; theme must not revert to `StyleColorsDark`.
