# Seamless tiles by filling a white hole

Can an image-to-image model make a tile seamless if we simply paint the part to redraw white and ask
it to fill the white in? This branch is three experiments that ask that question of FLUX.2 [klein] 9B,
on a fixed set of test patterns, with every intermediate picture kept.

The model is used as released: the distilled recipe (4 steps, guidance 1, one forward pass per step),
the masked picture given as an ordinary reference image, one prompt behind a system turn that tells
the model it is an inpainter. No mask channel, no inpainting head, no second prompt, no extra
guidance branch. The one change is the **seamless RoPE** of
[`src/flux2/torus.py`](src/flux2/torus.py): the output's tokens sit on a torus, so the model draws
the left edge next to the right edge and the top next to the bottom. There are two ways to get
there, and `--rope` picks one: `nearest` (the default) keeps the model's frequencies and moves every
displacement to its nearest periodic copy; `quantized` keeps the displacements and rounds every
frequency so that its rotation repeats after one tile. `--wrap False` turns that off and gives stock
FLUX.2 as the baseline.

## The three experiments

All three end the same way: a 1024 x 1024 picture with a white hole goes to the model. They differ in
where the picture comes from and how it is framed.

| | starts from | the model sees | question |
| :-- | :-- | :-- | :-- |
| `1_seam_fix` | an almost seamless tile | the tile rolled by half, a white cross of `--band` px over its seams | are small seam artifacts repaired? |
| `2_outpaint_frame` | the centre crop of the tile, `--border` px cut off every side: it does not repeat | the crop on a white canvas, so the hole is a frame around it | does outpainting the frame give a tile that repeats? |
| `3_outpaint_cross` | the same crop | the same canvas rolled by half, so the frame is a cross in the middle | does it help to move the hole to the middle? |

Rolling a tile means sliding a 1 x 1 window over a 2 x 2 board of copies: the same tile cut at a
different place. Rolled by half, the tile's own edges become two lines that cross in the middle of
the picture, where the model can draw across them.

**Two of the three are one run.** A frame of 64 px around a tile and a cross of 128 px through the
rolled tile are the same mask, so with `--band` = 2 x `--border` (the default) experiments 1 and 3
hand the model a byte-identical picture. The script generates it once and reports it twice: against
the original tile in experiment 1, against the crop in experiment 3. Experiment 2 is the other run,
and its input is that same picture before the roll, so 2 against 3 changes exactly one thing: where
the hole sits. Set `--border` on its own and 1 and 3 become separate runs.

## Run it

**Laptop: bring the patterns to one size.**

```bash
uv run python scripts/resize_tiles.py            # principled/test_patterns/** -> principled/tiles_1024/
rsync -av principled/tiles_1024 <server>:<repo>/principled/
```

Each pattern is resampled as one period of a repeat, so resizing neither adds a seam nor hides one,
and gets a plain ASCII name (`00_Original_Rose_Sample.png`). `tiles.csv` maps each back to its source.

**Server: run the experiments.** One process per GPU, the tiles split between them.

```bash
uv run python scripts/run_experiments.py                         # all tiles, all three experiments, all GPUs
uv run python scripts/run_experiments.py --only 00,03,19         # tiles by number ...
uv run python scripts/run_experiments.py --only rose,ivy         # ... or by name
uv run python scripts/run_experiments.py --tiles a.png,my/tiles  # any files, folders or globs
uv run python scripts/run_experiments.py --experiments 1         # 1, 2, 3 or the full names
uv run python scripts/run_experiments.py --band 64 --run_name band64
```

| option | default | |
| :-- | :-- | :-- |
| `--band` | `128` | width of the white cross in experiment 1 |
| `--border` | `band / 2` | width of the frame cut off and painted white in experiments 2 and 3 |
| `--prompt` | the fill prompt in [`experiments.py`](src/seamless/experiments.py) | the instruction, or a `.txt` file holding it |
| `--system_prompt` | the inpainter prompt, same file | the text encoder's system turn, in front of the prompt. `none` is the bare prompt |
| `--guidance` | `1.0` | 1 is the distilled recipe. Any other value is real CFG against the empty prompt: two passes per step |
| `--num_steps` | `4` | |
| `--seed` | `0` | |
| `--wrap` | `True` | the seamless RoPE and the circular decode. `False` is stock FLUX.2 |
| `--rope` | `nearest` | how the RoPE is made periodic: `nearest` copy of the displacement, or `quantized` frequencies (see `torus.py`) |
| `--unanchor_text` | `False` | the text no longer marks an origin on the torus (see `torus.py`) |
| `--model_name` | `flux.2-klein-9b` | any klein model |
| `--gpus` | all | physical ids, `0,1,2,3`; otherwise `CUDA_VISIBLE_DEVICES`, otherwise every GPU |
| `--run_name` | `<date>_<time>_band<band>` | the folder under `output/`; an existing run is never overwritten |

Bands and borders that are multiples of 32 and 16 px keep the hole's edges on the model's 16 px
token grid.

The system turn is not decoration. With the fill prompt alone the instruction does not reliably
take: some tiles come back with the white cross untouched, or with one white band left across the
middle. `--system_prompt none` reproduces that.

**Without a GPU.** Two ways to check a run before spending GPU time on it:

```bash
uv run python scripts/run_experiments.py --dry_run            # the masks and model inputs, nothing generated
uv run python scripts/run_experiments.py --toy --only 0,1     # random toy weights on the CPU: noise out, all code run
uv run python scripts/torus_selftest.py                       # the wrapped attention against its definition
```

## What comes out

```
output/<run_name>/
    index.html                  every tile x experiment: thumbnail, numbers, links
    summary.csv                 the same numbers, one row per job
    config.json                 everything needed to repeat the run
    logs/gpu<k>.log
    1_seam_fix/<tile>/
        1_original.png          the tile
        2_original_2x2.png      ... repeated 2 x 2
        3_rolled.png            rolled by half: the seams cross in the middle
        4_masked.png            the cross painted white          <- what the model is given
        5_generated.png         what the model returns
        6_unrolled.png          rolled back                      <- the result
        7_unrolled_2x2.png      ... repeated 2 x 2
        sheet.png               all of the above in one picture
        thumb.jpg  run.json
    2_outpaint_frame/<tile>/    1_original  2_original_2x2  3_crop  4_crop_2x2  5_padded  6_generated  7_generated_2x2
    3_outpaint_cross/<tile>/    1_original  2_original_2x2  3_crop  4_crop_2x2  5_padded  6_rolled  7_generated  8_unrolled  9_unrolled_2x2
```

`sheet.png` holds every step at its own pixel size, nothing resampled, so zooming into the sheet is
zooming into the step. What is drawn on it sits in the margins, never on a picture:

- **red arrows**: a seam, the line where the tile's own edges meet. Through the middle of a rolled
  picture; where the copies touch in a 2 x 2.
- **blue arrows**: where the edges of the *generated* picture meet once it is rolled back. Nothing
  was masked there, so it shows whether the model's output closes up on itself.
- **amber bars**: the extent of the hole, i.e. what is new in the output.

A sheet is 15-25 MB and a full run of 21 tiles is about 2.5 GB. To look at one from the laptop,
copy it (`rsync -av <server>:<repo>/output/<run> .`) or serve it and open `http://localhost:8000`:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory output/<run>     # on the server
ssh -L 8000:localhost:8000 <server>                                      # on the laptop
```

### The numbers

| | |
| :-- | :-- |
| `seam_before`, `seam_after` | seam jump of the tile the experiment starts from, and of the result |
| `wrap_after` | the same measure where the generated picture's own edges meet (rolled experiments) |
| `kept_psnr` | dB between the model's input and its output outside the hole: how much it changed what it was told to keep |

A *seam jump* is the mean absolute difference across a line, divided by the median of the same over
every parallel line of the picture; 1 means the line is like any other. It sees a hard cut. It does
not see a join that is smooth but wrong (a stem that bends, a motif that changes shape), which is
what the sheets are for.

## Code

```
scripts/resize_tiles.py       step 0: patterns of any size -> 1024 x 1024
scripts/run_experiments.py    step 1: jobs over GPUs, then the report
scripts/torus_selftest.py     the method checked on toy weights
src/seamless/tiles.py         roll, mask, crop, pad, repeat, seam jump        (numpy, PIL)
src/seamless/experiments.py   the three experiments as the pictures they make
src/seamless/sheet.py         the comparison sheet
src/seamless/klein.py         FLUX.2 klein image-to-image on the torus        (the only GPU code)
src/seamless/worker.py        one GPU's share of a run
src/seamless/report.py        index.html and summary.csv
src/flux2/torus.py            the seamless RoPE: attention with the nearest periodic copy
src/flux2/                    otherwise the FLUX.2 reference implementation, unchanged
```

## Licence

The code in `src/flux2` other than `torus.py` is Black Forest Labs' FLUX.2 inference code
([LICENSE.md](LICENSE.md)). The FLUX.2 [klein] 9B weights are under the
[FLUX Non-Commercial License](model_licenses/LICENSE-FLUX-NON-COMMERICAL).
