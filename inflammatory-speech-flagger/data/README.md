# Dataset sources

Use **one** of these. Option 1 is used by default in `notebook.ipynb`.

## Option 1 (recommended): AfriHate
- Link: https://huggingface.co/datasets/afrihate/afrihate
- Ships actual tweet text (not just IDs) across 15 African languages, annotated `Hate` / `Abusive` /
  `Normal` by native speakers. The notebook uses the 4 Nigerian-relevant configs: Nigerian Pidgin
  (`pcm`), Hausa (`hau`), Igbo (`ibo`), Yorùbá (`yor`).
- **Gated dataset** — requires a free Hugging Face account, accepting the dataset's access
  conditions on its page, and a read-access token. Full walkthrough is in `notebook.ipynb` Section 2.
  ```python
  from huggingface_hub import login
  login()  # paste your token when prompted
  from datasets import load_dataset
  ds = load_dataset('afrihate/afrihate', 'pcm')  # or 'hau', 'ibo', 'yor'
  ```
- Paper (for citing in your README/methodology): "AfriHate: A Multilingual Collection of Hate
  Speech and Abusive Language Datasets for African Languages" — https://arxiv.org/html/2501.08284v1

## Option 2 (was originally considered, has a blocker): NaijaHate
- Link: https://huggingface.co/datasets/worldbank/NaijaHate
- ~36,000 Nigerian tweets, well-annotated — **but this dataset only ships `tweet_id` + labels, not
  the actual tweet text**, because Twitter/X's terms don't allow redistributing raw tweet content.
  To use it you'd need to re-fetch each tweet's text through the paid X API, and many tweets from
  the 2007–2023 collection window are likely deleted or suspended by now. Not practical on a
  capstone deadline — documented here so you know why it was dropped, in case you want to mention
  it as a limitation ("considered but data access was infeasible in the timeframe").

## Option 3: HERDPhobia (narrower, ethnic-targeted hate speech)
- Paper: https://arxiv.org/pdf/2211.15262
- Focused specifically on hate speech against the Fulani ethnic group in Nigeria, in English,
  Nigerian Pidgin, and Hausa. Good supplementary data, not a full replacement for Option 1.

## Option 3: HERDPhobia (narrower, ethnic-targeted hate speech)
- Paper: https://arxiv.org/pdf/2211.15262
- Focused specifically on hate speech against the Fulani ethnic group in Nigeria, in English,
  Nigerian Pidgin, and Hausa. Good supplementary data, not a full replacement for Option 1.

## Option 4: Hand-labelled data (fallback / supplement)
If none of the above feel like a good enough fit, or your team wants Nigeria-specific slang not well
covered above, hand-label your own sample:
1. Collect ~300–500 real or representative posts (respect each platform's terms of service).
2. Put them in a spreadsheet with columns: `text, label` (label = 1 inflammatory, 0 not).
3. Have at least 2 people label independently and resolve disagreements — this is worth mentioning
   in your README as it strengthens your evaluation methodology.
4. Export as `labelled_posts.csv`, upload to Colab, and use Option B in Section 2 of the notebook.
