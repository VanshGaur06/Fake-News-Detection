# LIAR dataset setup

Dataset source: <https://github.com/tfs4/liar_dataset>

Run `python src/download_dataset.py` to download only `train.tsv`, `valid.tsv`, and `test.tsv` into `data/liar/`. The downloader validates the 14-column TSV format and expected LIAR labels. The raw files are ignored by Git. To set them up manually, put the three original split files in `data/liar/`.
