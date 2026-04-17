---
name: bisect-precision
description: Locate which pypto or simpler commit introduced a precision regression. Diffs generated code to determine which repo to bisect, then runs git bisect with automated compile-and-test.
---

# Bisect Precision Regression

Locate the exact commit that introduced a precision regression in a pypto-lib
example by binary-searching through pypto (compiler) or simpler (runtime) history.

## Prerequisites

- The example script must have been passing before and is now failing precision.
- The user knows (or MEMORY.md records) a **known-good commit** for both pypto and
  simpler. If not available, ask the user.
- The example must be runnable via `python <script> -p <platform> -d <device>`.
- pypto and simpler repos are locally available (typically `../../pypto` and
  `/data/.../simpler` or resolved from SIMPLER_ROOT / the pypto submodule).

## Workflow

### Phase 0: Gather Context

1. Ask the user which example script is failing (if not already clear).
2. Read `memory/MEMORY.md` for known-good commits. If none recorded, ask the user
   for the last-known-good commits of pypto and simpler.
3. Record the **current HEAD** of both repos as the known-bad commits.
4. Determine the run command. Typical pattern:
   ```bash
   source run.sh  # or manually export env vars
   python examples/models/qwen3/<script>.py -p a2a3 -d <device>
   ```

### Phase 1: Triage — Compiler vs Runtime

Compare the **generated code** from a known-good build_output against the current
(bad) build_output to decide which repo to bisect.

#### Step 1a: Identify build outputs

List `build_output/` directories and identify:
- **Bad build**: the most recent build output (from the failing run).
- **Good build**: a build output generated when precision was passing. The user may
  know which timestamp corresponds to the good run. If no good build output exists
  on disk, you will need to generate one (see Step 1b).

#### Step 1b: Generate a good build output (if needed)

If no known-good build_output is available:

```bash
# Save current HEADs
PYPTO_BAD=$(git -C <pypto_dir> rev-parse HEAD)
SIMPLER_BAD=$(git -C <simpler_dir> rev-parse HEAD)

# Checkout known-good commits
git -C <pypto_dir> checkout <pypto_good_commit>
pip install --no-build-isolation -e <pypto_dir>
git -C <simpler_dir> checkout <simpler_good_commit>
pip install --no-build-isolation -e <simpler_dir>

# Run the example to generate a good build_output
python <script> -p <platform> -d <device>

# The newest build_output/<ProgramName>_<timestamp> is the good build.
GOOD_BUILD=$(ls -td build_output/<ProgramName>_* | head -1)

# Restore bad commits
git -C <pypto_dir> checkout $PYPTO_BAD
pip install --no-build-isolation -e <pypto_dir>
git -C <simpler_dir> checkout $SIMPLER_BAD
pip install --no-build-isolation -e <simpler_dir>
```

#### Step 1c: Diff generated code

```bash
diff -rq $GOOD_BUILD/kernels/  $BAD_BUILD/kernels/
diff -rq $GOOD_BUILD/ptoas/    $BAD_BUILD/ptoas/
diff -rq $GOOD_BUILD/orchestration/ $BAD_BUILD/orchestration/
```

**Interpretation:**

| Diff result | Meaning | Action |
|---|---|---|
| **Files differ** in kernels/, ptoas/, or orchestration/ | Compiler (pypto) generated different code | Bisect **pypto**, pin simpler at known-good |
| **No differences** (or only timestamps / .o binaries differ) | Same generated code, runtime behavior changed | Bisect **simpler**, pin pypto at known-good |

If both have differences, start with pypto (compiler changes are more common).

Report the triage result to the user before proceeding.

### Phase 2: Git Bisect

#### Determine bisect parameters

```bash
# Example for pypto bisect:
REPO_DIR=<pypto_dir>
GOOD_COMMIT=<known_good>
BAD_COMMIT=<current_HEAD>

# Count commits in range
git -C $REPO_DIR log --oneline $GOOD_COMMIT..$BAD_COMMIT | wc -l
```

Report the commit count to the user (~6 bisect steps for 64 commits).

#### Pin the other repo at known-good

If bisecting pypto, pin simpler at its known-good commit (and vice versa):

```bash
git -C <other_repo> checkout <known_good_commit>
pip install --no-build-isolation -e <other_repo>
```

#### Run the bisect

```bash
cd $REPO_DIR
git bisect start
git bisect bad $BAD_COMMIT
git bisect good $GOOD_COMMIT
```

For each bisect step:

1. **Install the current checkout:**
   ```bash
   pip install --no-build-isolation -e .
   ```
   If install fails (e.g. incompatible API change), mark as skip:
   ```bash
   git bisect skip
   ```

2. **Run the example:**
   ```bash
   cd <pypto_lib_dir>
   python <script> -p <platform> -d <device>
   ```

3. **Check result:**
   - If output contains `PASS` → `git bisect good`
   - If output contains `FAIL` or non-zero exit → `git bisect bad`
   - If crash / unrelated error → `git bisect skip`

4. **Report progress** to the user after each step:
   ```
   Step N/~M: <commit_short> — <commit_message> → good/bad/skip
   ```

5. Repeat until git bisect identifies the first bad commit.

#### Bisect complete

When `git bisect` prints the first bad commit:

```bash
# Save the result
CULPRIT=$(git bisect view --oneline)
git bisect reset
```

### Phase 3: Report

Present to the user:

1. **Triage result**: compiler (pypto) or runtime (simpler) regression.
2. **First bad commit**: hash, message, author, date.
3. **What changed**: `git show <commit> --stat` summary.
4. **Diff highlights**: key changes that likely affect precision (look for dtype
   changes, cast operations, memory layout changes, optimization pass changes).
5. **Restore state**: confirm both repos are back to their original HEAD.

```bash
git -C <pypto_dir> checkout <original_pypto_HEAD>
pip install --no-build-isolation -e <pypto_dir>
git -C <simpler_dir> checkout <original_simpler_HEAD>
pip install --no-build-isolation -e <simpler_dir>
```

## Important Notes

- **Always restore repos** to their original HEAD after bisect, regardless of
  outcome.
- **Each `pip install`** of pypto takes ~30s-2min (C++ recompilation). Simpler is
  similar. Factor this into time expectations.
- **Environment variables**: make sure PTOAS_ROOT, PTO_ISA_ROOT, ASCEND_HOME_PATH,
  and runtime pool sizes (PTO2_RING_*) are set. Check `run.sh` in the project root.
- **Build output comparison** ignores `.o` / `.so` files (binary artifacts) — only
  compare `.cpp` and `.pto` source files.
- If the example uses `--clear-cache`, include that flag during bisect to avoid
  stale cached data affecting results.
- If `pip install` fails on a specific commit (e.g., API breaking change), use
  `git bisect skip` — git will try adjacent commits.

## Asking the User

At any point, if information is missing:

| Missing info | What to ask |
|---|---|
| Known-good pypto commit | "What is the last pypto commit where precision was passing?" |
| Known-good simpler commit | "What is the last simpler commit where precision was passing?" |
| Which build_output is good | "Which build_output timestamp was from a passing run?" |
| Platform / device | "Which platform and device ID should I use? (e.g., `-p a2a3 -d 5`)" |
| Script to run | "Which example script is failing? (e.g., `qwen3_32b_prefill_scope2.py`)" |
