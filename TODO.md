# TODO

## Future Features

- [x] Add DDSP effects from [magenta/ddsp/effects.py](https://github.com/magenta/ddsp/blob/main/ddsp/effects.py) as graph editor elements:
  - [x] **Reverb** — convolutional (FIR) reverb; FFT convolve dry audio with impulse response; params: `reverb_length`, `add_dry`, optional external/trainable IR input
  - [x] **ExpDecayReverb** — IR parameterized as exponential decay of white noise; params: `gain`, `decay`, `reverb_length`, `add_dry`
  - [x] **FilteredNoiseReverb** — IR synthesized via filtered noise; params: `magnitudes` `[time × filter_banks]`, `window_size`, `reverb_length`, `add_dry`
  - [x] **FIRFilter** — linear time-varying (LTV-FIR) frequency-domain filter; params: `magnitudes` `[time × filter_banks]`, `window_size`
  - [x] **ModDelay** — modulated delay for chorus/flanger/vibrato; params: `center_ms`, `depth_ms`, `gain`, `phase`, `add_dry`
- [x] add lstm and rnn
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
5) macros of an element should be saved with it in user library


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

- [ ] fix rms levels not showing after groups and from group input, and after deletion of group between audio in->out
- [ ] allow creating group of one
- [ ] allow wrong connections and eg insertion of wrong element, cut sound and show warning to change shape parameter if necessary

- [ ] Big v1 rave: noise path should also do Math x1 - 5 → Sigmoid → Math 2 * x1^2.3 + 1e-7?
- [ ] add bias option to elements
- [ ] compactness not ready
- [ ] input to latent or pca?
- [ ] load last checkpoint of rave: https://github.com/acids-ircam/RAVE/discussions/82

- [ ] trained boxes should store a list of checkpoints to select from, with "best" one too. 

- [ ] Remove TCN because was reproduced as a group (but macros needed for convenience)
- [ ] remove linear because it's conv1d with stride=kernel=dilation=1

- [ ] Reproduce NAM neural amp modeler
- [ ] neural waveshaping synthesis
- [ ] latent jamming: 
  - https://github.com/devstermarts/PD-Latent-Jamming
  - Błażej Kotowski network bending: https://www.youtube.com/watch?v=RXl4NwG5go0&t=1s
  - Brave: 
    - https://www.youtube.com/watch?v=0HugWkdesgw
    - https://github.com/danielmanz17/Brave 
  - Latent jamming / prior as a partner. A second net (GRU is the live-cheap one) predicts the next z so you jam with a continuation of the corpus, not a clone of the input. MSPrior (https://github.com/caillonantoine/msprior), Caillon’s semantic-hand demo (https://caillonantoine.github.io/2023/05/16/semantic-control.html). Your audio becomes a suggestion, not a waveform to copy.
- Latent granular resynthesis: https://arxiv.org/abs/2507.19202
