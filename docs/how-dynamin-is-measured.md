# How dynamin is measured

A plain-language description of what the program actually does with your
data. It is written so that a reader without a technical background can
follow it; the technical details, with references into the code, live in
the project README.

The program receives a microscopy video and a list of places where
clathrin-coated pits sit on the cell membrane. For each place it answers
a single question: **how much dynamin light is shining there, at that
moment?**

---

## The big picture: what happens when you hand it data

The input is two things:

- a **video** — a TIFF file: hundreds of frames, each a grid of
  brightness values (pixels), with the dynamin channel inside;
- a **list of coordinates** — for every point, a frame number and a
  position where the clathrin channel says a pit is. The program never
  finds pits itself; you supply the positions, typically from a tracking
  program.

Before measuring, the program looks at the first frame and runs two
**input checks**: are these by mistake raw SIM frames with illumination
stripes (those must be averaged before any measurement)? And do the two
channels sit on top of each other (is the dynamin image shifted relative
to clathrin)? If something looks wrong, the program **warns but
continues** — the decision stays with you.

Then it takes the points one by one. For each point it cuts a small
**23 × 23 pixel window** around the given position, runs two fits inside
it (described below), and appends one row of results to your table:

| column | meaning |
|---|---|
| `dnm_A` | **The main result:** the dynamin brightness at that spot — the height of the "hill of light" above the local background. |
| `dnm_c` | The local background level (how much the surroundings glow on their own). |
| `dnm_A_pstd` | The uncertainty of the main result — how much plus/minus you can trust it. |
| `dnm_pval`, `dnm_signif` | The verdict: how easily a value like this could arise from pure noise, and a shortened yes/no. |
| `dnm_x`, `dnm_y` | The refined position, if the fit was allowed to shift the centre (see question 1). |
| `dnm_valid` | Whether the measurement ran at all (0 only for points too close to the image edge). |

Nothing else happens to the video. The rest of this document walks
through the five questions from the original assignment.

---

## 1. How does the program search for the dynamin blob around a given position?

**Short answer: it doesn't search.** It measures exactly where clathrin
points — and only allows the fit a small, well-guarded correction of the
position.

Imagine measuring the brightness of a faint lamp on a wall while a
colleague points a finger: "there." You can do two things: press your
light meter exactly onto that spot and read the value, or glance around
the immediate neighbourhood in case the lamp is a touch to the side.
The program does both, in this order:

1. **Fit 1 — position locked.** At the given coordinate it computes how
   tall a "hill of light" sits there. The position may not move by a
   single pixel. The result is an honest answer to "how much light is
   *right here*" — even if that answer is zero.
2. **Fit 2 — position free to shift.** The same computation runs again,
   but the centre of the hill may adjust itself. If the dynamin spot
   sits slightly off the clathrin spot, this fit finds it.

The shifted result is accepted **only if it passes both conditions at
once**: it moved only a little (roughly up to 4 pixels for our data;
two independent safety limits guard this) *and* it found more light.
Otherwise the locked-position result stands.

Why the caution? There is little dynamin on a pit, and its signal drowns
in noise. If the program simply "searched," it would happily latch onto
the nearest noise grain at weak spots and report light where there is
none. The locked fit prevents that — and the shifted fit is trusted only
when the shift is small and the result demonstrably better.

> **What clathrin is for.** The clathrin channel serves as a
> **pointer** — it is bright and reliable, so it says *where* to
> measure. The measurement itself happens exclusively in the dynamin
> channel. That is why the two channels must be registered onto each
> other; the program checks this at start-up.

---

## 2. How exactly does the Gaussian fitting work, and which value becomes the intensity?

**Short answer: the result is the height of the hill of light above the
local background — nothing more, nothing less.**

A single glowing molecular spot does not appear in a microscope as a
point but as a small blurred **hill of light** — the physics of the
microscope always smears it into the same bell shape (a Gaussian
curve). "Fitting" then means: take a mathematical hill and tune its
properties until it matches the actual pixel values in the 23 × 23
window as closely as possible.

A rough sketch of one window, seen in cross-section:

```
 brightness
     │            ___
     │           /   \        ← the fitted hill (Gaussian)
     │          /     \
     │    A →  |       |        A = height above the floor
     │         /       \            = THE RESULT
     │  ....../.........\......  ← c = the floor (local background)
     │  ▂▃▂▄▅▆▇█▇▆▅▄▂▃▂▃▂       ← measured pixels
     └──────────────────────── position across the window
```

The hill has four properties, but the program may only adjust some:

- **Height (A)** — always adjusted. **This is the resulting intensity.**
- **Floor (c)** — always adjusted, together with the height. Because of
  this, the result is automatically "above background": the glow of the
  surroundings is not counted into it.
- **Position (x, y)** — locked in fit 1, free in fit 2 (see question 1).
- **Width (σ)** — *never* adjusted. It is measured beforehand from the
  whole dataset (see question 5), because the microscope's physics fixes
  it for all measurements alike. If the width were free, the fit could
  "inflate" a wide flat hill out of pure noise.

So the intensity is **not** the value of the brightest pixel and **not**
the summed volume of the hill — it is the distance from the floor to the
peak, in raw camera units.

Alongside the height, the program also returns an **uncertainty** (how
far the height might be off, given the noise in the window) and a
**p-value** — a number between 0 and 1 saying how easily an equally tall
hill would arise by pure chance. A small p-value means "this is almost
certainly not noise."

---

## 3. What happens when the fit fails?

**The most important point first: "no dynamin" is not a failure.** An
empty spot returns a number close to zero — not an error.

This is the most common misunderstanding. When there is no dynamin at
the measured spot, the fit runs perfectly normally — it simply yields a
hill with a height around zero (possibly slightly negative, which is the
natural face of noise). The p-value then says: "not significant."
**Measuring and judging are two separate steps** — the program always
measures honestly and leaves the verdict to the statistics.

Genuine trouble arises in exactly three situations, each with defined
behaviour:

| situation | what the program does |
|---|---|
| The point is too close to the image edge | The 23 × 23 window does not fit inside the frame. The point is skipped and marked `dnm_valid = 0`; its results are empty (NaN). In our data this affects fewer than 0.1 % of points. |
| The shifted fit "ran away" | If fit 2 drifts too far, or lands on an absurdly tall hill, its result is discarded and the locked-position fit 1 is quietly used instead. You never lose the measurement. |
| Nonsensical input coordinates | Frame numbers out of range stop the run with a clear error; a non-numeric position yields an empty result. The program never crashes mid-run over a single bad point. |

> **Why negative numbers are fine.** At a spot with no signal, noise
> fluctuates randomly above and below zero. If the program forbade
> negative heights, averages over empty places would come out
> artificially positive — and a faint real signal could no longer be
> told apart from them. A negative value at an insignificant point is a
> sign of honest measurement, not of a bug.

---

## 4. How is the resulting intensity normalised?

**Short answer: it isn't — deliberately.** The original cmeAnalysis does
the same.

The result is in **raw camera units** — exactly the numbers the camera
wrote into the file. The only thing "subtracted" is the local
background, and even that is not a separate adjustment but part of the
measurement's construction: the height of the hill is measured from the
floor from the start (see the sketch in question 2).

What is *not* done to the number:

- no camera corrections (offset, gain),
- no correction for fluorophore fading over the movie (photobleaching),
- no conversion to photons or molecule counts,
- no scaling between movies.

The reason is simple: every such adjustment is a *decision about
interpreting the data*, and that belongs in the hands of the person
analysing it — not inside a measuring function where it would happen
invisibly. The program therefore returns raw numbers and leaves
adjustments as conscious, separate steps.

> **One concrete warning.** The package does include a tool for
> equalising brightness across movies (ported from cmeAnalysis). For
> this dataset we advise **against** switching it on: the differences
> between movies here are largely biological — the dynamin knockdown
> took hold with different strength in each cell. Equalising the movies
> would erase exactly the effect being measured.

---

## 5. Does the computation use statistics from the whole video, or from several videos?

**Short answer: the number itself is born in the small window. Only two
global pieces of information enter — and it is worth knowing which.**

It helps to distinguish two floors of the program:

### Floor 1 — measurement (one number for one point)

| what enters | where it comes from | does it affect the number? |
|---|---|---|
| the 23 × 23 window of pixels | a single place in a single frame | **yes — this is the whole measurement** |
| the hill width σ | **all 23 movies of the dataset** — measured once from thousands of bright spots, then held fixed for every measurement | yes, but only mildly (verified: ±9 % in σ moves the result by ~2 %) |
| the brightest and darkest pixel of the frame | the whole current frame | no — it only serves as a guard against absurd fits |

### Floor 2 — classification ("does this track have dynamin?")

The follow-up step, which decides from the measurements along a track
whether the track is dynamin-positive, does use whole-movie statistics —
and it has to: to recognise what counts as "significantly above
background," it needs to know what the background of *this particular
movie* looks like. From 16 frames spread across the movie it measures
the brightness distribution at empty places inside the cell (away from
pits) and the rate of random "false spots." Every track is then tested
against these, with strictness proportional to its length.

> **Summary in one breath.** The number for one point depends on the
> window around it and on one constant calibrated from the whole
> dataset. Whole-movie statistics do not enter the *number* — they enter
> the *verdict* on whether the number is significant. Across different
> videos nothing is shared except σ: every movie has its own background
> and its own verdicts.

---

## What the program deliberately does not do

- **It does not find pits.** You supply the coordinates; it only
  measures at them.
- **It does not track.** Linking points into tracks over time happens
  before it.
- **It does not work with time.** A frame number is just an index to it;
  it does not know the interval between frames, so it cannot produce
  seconds, lifetimes or rates.
- **It does not normalise or correct** — see question 4.
- **It does not decide for you.** It returns numbers and p-values;
  thresholds and interpretation stay in the analyst's hands.

---

## A small glossary

| term | meaning |
|---|---|
| pixel | One cell of the image grid; it carries a single number — how much light fell on it. |
| hill (Gaussian) | The shape into which the microscope smears every glowing spot: a bell curve described by position, height and width. |
| σ (sigma) | The width of the hill. A property of the microscope and the light's wavelength, not of the sample — which is why it is measured once and then held fixed. |
| background | Light that glows even where no spot is — stray molecules, reflections, camera electronics. The model captures it as the "floor" under the hill. |
| p-value | The probability that an equally strong result would arise from pure chance in noise. The smaller, the more believable the signal. |
| NaN | "Not a Number" — the marker for an empty result. In our outputs it means one thing only: the point lay too close to the image edge. |
| master / slave channel | cmeAnalysis jargon: the master is the bright pointing channel (clathrin), the slave is the measured channel (dynamin). |

---

*CMEpython — a Python port of the dynamin intensity measurement from
cmeAnalysis (Danuser Lab, GPL-3.0). The port's fidelity is verified
directly against the original program: on 874 samples of real data the
results agree to 7 or more decimal places.*
