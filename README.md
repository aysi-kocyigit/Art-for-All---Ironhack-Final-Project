# Art-for-All---Ironhack-Final-Project

Art for All is a Streamlit application that helps viewers articulate why an artwork feels the way it does.
Upload an image, describe your emotions, and receive:

* A structured formal visual analysis
* Color EDA from pixels (dominant palette, RGB and brightness distributions, warm/cool balance)
* A visual–emotion write-up that ties concrete visual elements to feelings
* An art-historical context suggestion with movement matches and transparent reasons

---

## Features

**Formal visual analysis (Vision LLM)**

* Produces a consistent, professional write-up with these sections:

  * Scene & Subject Matter
  * Composition & Structure
  * Technique & Medium
  * Color & Light Analysis
  * Mood & Atmospheric Qualities

**Color data analysis (no external CV deps)**

* Dominant colors via PIL quantization, named using CSS color names
* RGB histograms and brightness histogram
* Warm/Cool/Neutral split and a radial “Top-5 palette” view

**Emotion NLP**

* Maps free-text responses to canonical emotion keywords
* Uses Hugging Face `j-hartmann/emotion-english-distilroberta-base`

**Movement ranking (transparent heuristic)**

* Techniques (55%) + Color-derived moods (30%) + User-emotion overlap (10%) + Mood cues (5%)
* Bonuses for style hints parsed from the vision analysis and for palette temperature alignment

**Readable explanations**

* Emotional & psychological impact of visual elements
* Art-historical context with movement overview, artists, and “what to look for” guidance

**Clean UI**

* Four tabs: Formal Analysis, Color Data Analysis, Emotional Response Analysis, Art Historical Context
* Light, minimal “gallery” theme via `config.toml`

---

## Project Structure

```
.
├─ app.py
├─ vision_analysis.py
├─ emotion_processing.py
├─ color_utils.py
├─ utils.py
├─ recommendations.py
├─ data/
│  ├─ art_movements.json
│  ├─ art_techniques.json
│  └─ color_emotions.json
├─ config.toml
└─ README.md
```

* `app.py` — Streamlit UI and orchestration
* `vision_analysis.py` — OpenAI Vision prompt and image encoding
* `emotion_processing.py` — Emotion NLP, analysis parsers, movement ranker, LLM explainers
* `color_utils.py` — Dominant color extraction and CSS color naming
* `utils.py` — Cached data loading, formatting helpers
* `data/` — Curated knowledge bases (movements, techniques, color–emotion)

---

## Setup

**Python:** 3.11 recommended

Create and activate an environment:

```bash
conda create -n art-bridge python=3.11 -y
conda activate art-bridge
```

Install dependencies:

```bash
pip install streamlit pillow pandas numpy matplotlib seaborn python-dotenv transformers openai
```

Set your OpenAI API key in a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

Run the app:

```bash
streamlit run app.py
```

Open the local URL, upload an artwork image, describe your emotional response, and click “Analyze.”

---

## Configuration

Theme (`config.toml`):

```toml
[theme]
base = "light"
primaryColor = "#6E59A5"
backgroundColor = "#FAFAF8"
secondaryBackgroundColor = "#EEEDE8"
textColor = "#1B1B1F"
font = "sans serif"
```

Models used:

* Vision and text generation: `gpt-4o`, `gpt-4o-mini`
* Emotion classifier: `j-hartmann/emotion-english-distilroberta-base`

---

## How It Works

1. **Input**
   User uploads an image and provides a brief emotion description (free text or tags).

2. **Formal analysis (vision LLM)**
   A strict rubric prompts the model to produce a structured, professional write-up.

3. **Parsing**
   Regex and vocabularies extract techniques, color tokens, mood words, temperature/contrast, and style hints.

4. **Color EDA**
   Dominant colors are computed from pixels; charts display palette share, RGB distributions, brightness, and temperature split.

5. **Emotion NLP**
   Text is mapped to canonical emotion keywords using a compact Hugging Face model.

6. **Movement ranking**
   Composite score: 55% techniques, 30% color-mood, 10% emotion overlap, 5% mood cues; plus bonuses for style hints and temperature.

7. **Explanations**
   Two targeted prompts generate:

   * an emotional/visual impact analysis, and
   * a deeper art-historical context write-up.
     The code tries the OpenAI Responses API and falls back to Chat Completions for compatibility.

8. **Presentation**
   Streamlit renders four tabs with metrics, tables, and plots.

---

## Data Files

* `data/art_movements.json` — movement characteristics, emotions, artists, palette cues, techniques
* `data/color_emotions.json` — color and shade entries with emotion lists and short psychology blurbs
* `data/art_techniques.json` — technique descriptions, emotional impact, example artists

The files are intentionally compact and easy to extend.

---

## Troubleshooting

**OpenAI SDK mismatch**
If you encounter `TypeError: Responses.create() got an unexpected keyword argument ...`, the code will fall back to `chat.completions`. Ensure:

* `OPENAI_API_KEY` is set,
* a recent `openai` package is installed.

**`KeyError: 'Hex Code'` in the color table**
Call `display_color_statistics` with the original `dominant_colors` structure from `extract_dominant_colors`. Do not rename or drop the `Hex Code` column in the DataFrame.

**Slow first run for Transformers**
The model download occurs on first use and is cached afterwards.

**Vision analysis errors**
Check the `.env` key, model availability, and that images are readable (the app re-encodes to JPEG RGB internally).

---

## Roadmap

* Museum API integrations (Met, WikiArt) for examples and citations
* Multilingual UX and prompts
* Side-by-side artwork comparisons
* Expanded and validated movement/technique knowledge bases
* Optional offline-lean mode with reduced features

---

## License

MIT License (add a `LICENSE` file if missing).

---

## Citation

Kocyigit, A. (2025). Art for ALl  — a Streamlit app linking viewer emotions to visual analysis and art-historical context.

---

## Contact

For questions or ideas, please open an issue in this repository.
