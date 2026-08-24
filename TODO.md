# TODO

## Future Features

- [ ] Add per-element visualization graphs showing cumulative sound transformation up to that element (e.g., transfer graph, frequency response, and related analysis views). Both left and right channels should be shown on the same plots.
- [ ] Add gain control support to the activation function element and tcn. should control slope of function.
- [ ] Add knob inputs.
- [ ] Add XY trackpad.
- [ ] Add DDSP effects from [magenta/ddsp/effects.py](https://github.com/magenta/ddsp/blob/main/ddsp/effects.py) as graph editor elements:
  - [ ] **Reverb** — convolutional (FIR) reverb; FFT convolve dry audio with impulse response; params: `reverb_length`, `add_dry`, optional external/trainable IR input
  - [ ] **ExpDecayReverb** — IR parameterized as exponential decay of white noise; params: `gain`, `decay`, `reverb_length`, `add_dry`
  - [ ] **FilteredNoiseReverb** — IR synthesized via filtered noise; params: `magnitudes` `[time × filter_banks]`, `window_size`, `reverb_length`, `add_dry`
  - [ ] **FIRFilter** — linear time-varying (LTV-FIR) frequency-domain filter; params: `magnitudes` `[time × filter_banks]`, `window_size`
  - [ ] **ModDelay** — modulated delay for chorus/flanger/vibrato; params: `center_ms`, `depth_ms`, `gain`, `phase`, `add_dry`
- [ ] Find good compressor DDSP to add too.
- [ ] Implement element set for RAVE.
- [ ] Implement element set for MRT2.
- [ ] add positional encoding
- [ ] add diffusion process? (IRCAM AFTER?)

