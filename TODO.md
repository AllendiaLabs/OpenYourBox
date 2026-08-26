# TODO

## Future Features

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
- [ ] add dropout
- [ ] check that bn reset resets to normal, not random, then remove seed

- [ ] read steinmetz frontiers: Differentiable black-box and gray-box modeling of nonlinear audio effects

- [ ] in addition to the graph editor view, add a performance view where users can choose to display parameters or controls, change their size, layout, etc
- [ ] implement preset management

- [ ] 1) when property/parameter, knob or xy trackpad (or a macro) is selected, delete key should reset to default value
2) if in a group, right click on them should allow the user to create a macro and assign to it (adds control to group box), or all similar parameters/controls in the group (not subgroups), or assign to existing macro
3) when assigned to macro, disable it so that users cannot desync values
4) add necessary parameters to group box, including trackpad and knobs, to control values from the group box


- [ ] check rf same as in rave paper