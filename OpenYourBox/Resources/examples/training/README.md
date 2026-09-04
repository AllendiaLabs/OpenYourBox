# Training examples

Loadable **graph templates** and **training configs** for architecture-agnostic training.

These are examples/templates, not Train modes. Open Train and use:

- **Example: mapping-style** — Conv1D + Activation with distinct input/target Data Loader feeds and an `mr_stft` Loss (`mapping-style-graph.xml` + `mapping-style-config.json`).
- **Example: reconstruction-style** — downsample Conv1D + Variational Bottleneck + ConvTranspose with same-data loader wiring, spectral + KL losses, and a two-stage schedule (`reconstruction-style-graph.xml` + `reconstruction-style-config.json`).

Bind Training Library audio onto Data Loader outputs after loading a graph; example configs do not store corpora.
