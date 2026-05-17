# Optional Strict Filtering Resources

This directory is optional.

The normal local PDF -> chunk pipeline does not require files here.

If you explicitly run:

```powershell
python scripts/run_pipeline.py --strict
```

then `scripts/pipeline/04_pes2o_filter_clean.py` requires:

```text
data/resources/unigram_freq.csv
```

That CSV is not shipped with this project. It should contain unigram frequency/probability data with columns such as:

```text
word,count
```

or:

```text
word,freq
```

Without this file, use the default non-strict local mode:

```powershell
python scripts/run_pipeline.py --skip-grobid --rebuild-chroma
```
