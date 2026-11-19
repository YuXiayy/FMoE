# Reference metric sources

These files are retained only for numerical equivalence testing:

- `audiosr/demo_comparison.py` is from the [AudioSR repository](https://github.com/haoheliu/versatile_audio_super_resolution), specifically its LSD calculation example.
- `nussl/bss_eval.py` is from the [nussl repository](https://github.com/nussl/nussl), specifically its scale-invariant BSS evaluation implementation.

`python evaluate.py verify-kernels` extracts only the relevant public functions and compares their numerical outputs with the vectorized implementation used by `../run_audio_metrics.py`. Runtime evaluation does not import these archived modules.

The source files remain subject to their upstream project licenses. Refer to the linked repositories for their license notices and version history.
