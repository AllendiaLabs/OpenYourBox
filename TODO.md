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

- [ ] read steinmetz frontiers: Differentiable black-box and gray-box modeling of nonlinear audio effects

- [ ] in addition to the graph editor view, add a performance view where users can choose to display parameters or controls, change their size, layout, etc
- [x] implement preset management and session undo/redo (`specs/008-preset-undo-history/`)

- [ ] 1) when property/parameter, knob or xy trackpad (or a macro) is selected, delete key should reset to default value
2) if in a group, right click on them should allow the user to create a macro and assign to it (adds control to group box), or all similar parameters/controls in the group (not subgroups), or assign to existing macro
3) when assigned to macro, disable it so that users cannot desync values
4) add necessary parameters to group box, including trackpad and knobs, to control values from the group box


- [ ] check rf same as in rave paper
- [x] update pin shapes with list when multiple copies in group
- [x] oyb should reproduce acids-ircam rave code more faithfully, starting with (`specs/009-rave-vae-parity/`):
  - [x] add softplus to vae and perform exactly the same parameterization/sampling
  - [x] kernel size of variational convs should be 5 (but add as bottleneck parameter for flexibility). and instead of 2 convs that see all input channels, use grouped head: Conv 1024→256, k=5, groups=2 (or equivalent channel split)
  - [x] PCA on μ in eval mode over a validation pass, use linear singular-value cumulative sum for r_f, and load compactness buffers onto the live bottleneck after Unfreeze.



- [ ] add depthwise to TCN. and to conv1d? what is it exactly? like groups?


- [ ] move parameters from boxes to new tab in right menu
- [ ] double click on group box to open it (it does not work anymore)
- [x] from user library or project structure, clicking on an element should open its property panel
- [x] from project structure, double clicking on a non-group element should go to that element (camera centered on it). double-clicking a group opens its inner canvas
- [ ] from project structure, dragging an element can move it within the structure, outside a group or in a group. during drag, highlight parent folder/group or new folder it will be added to. highlight by showing bounds of the group, ie a rectangle around its name and content.
- [ ] from user library, dragging to the canva should enable the user to add to project
- [ ] now it is very annoying to move a box, I must click on it many times before being able to drag it. clicking on it should directly select it and make it draggable, in one hold click
- [ ] when I move group boxes they glitch: they can change size, and the randomize button changes size too. the button should be in right property menu.
- [ ] only keep name and pins on boxes. the rest goes to property menu.
- [ ] selecting or clicking on a box should open its property menu 

- [x] deleting B in A->B->C should lead to A->C, not disconnected A C. B being a box, a selection of boxes, groups, etc. if connection cannot be made then do not connect (no need for warning in ui). make sure all edge cases are taken into account