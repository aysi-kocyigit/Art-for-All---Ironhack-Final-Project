# emotion_processing.py
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from transformers import pipeline
from openai import OpenAI
import openai

# ---------- Models ----------
# Emotion classifier for detecting emotions in user text
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# ---------- Lexicons & Helpers ----------
# Canonical emotions mapped to their variants/synonyms
CANON_EMOTIONS = {
    "peaceful": ["peaceful", "calm", "serene", "tranquil", "soothing"],
    "joyful": ["joyful", "happy", "cheerful", "optimistic"],
    "melancholy": ["melancholy", "sad", "sorrowful", "blue"],
    "anxious": ["anxious", "tense", "uneasy", "nervous"],
    "intense": ["intense", "powerful", "charged"],
    "dramatic": ["dramatic", "theatrical"],
    "contemplative": ["contemplative", "thoughtful", "pensive"],
    "mysterious": ["mysterious", "enigmatic", "dreamlike", "uncanny"],
    "energetic": ["energetic", "excited", "vibrant"],
    "angry": ["angry", "aggressive", "furious"],
    "overwhelming": ["overwhelming", "immersive"]
}

# Expanded color vocabulary for detecting color words in analysis
COLOR_WORDS = [
    # primaries / basics
    "red", "blue", "yellow", "green", "purple", "orange", "pink", "brown", "black", "white", "gray", "grey",
    # neutrals & metals
    "beige", "sand", "tan", "taupe", "ivory", "cream", "alabaster",
    "charcoal", "slate", "gunmetal", "pewter",
    "gold", "silver", "bronze", "copper",
    # oranges / reds extended
    "crimson", "burgundy", "maroon", "coral", "peach", "salmon", "rust", "terracotta", "amber", "mustard",
    # greens extended
    "sage", "olive", "chartreuse", "lime", "mint", "emerald", "forest", "khaki",
    # blues & cyans extended
    "sky", "azure", "cerulean", "cobalt", "ultramarine", "navy", "teal", "turquoise", "cyan",
    # purples & pinks extended
    "violet", "indigo", "lavender", "lilac", "mauve", "magenta", "fuchsia", "rose", "blush", "hot_pink",
    # meta-categories
    "pastel", "monochrome"
]

# Normalize shade → base color mapping
COLOR_ALIASES: Dict[str, str] = {
    # greys
    "grey": "gray", "charcoal": "gray", "slate": "gray", "gunmetal": "gray", "pewter": "gray",
    # whites
    "ivory": "white", "cream": "white", "alabaster": "white",
    # browns / earths
    "taupe": "brown", "tan": "brown", "sand": "beige", "khaki": "brown",
    "rust": "orange", "terracotta": "orange", "amber": "orange", "mustard": "yellow",
    "chocolate": "brown", "mahogany": "brown", "bronze": "brown", "copper": "brown",
    # reds
    "crimson": "red", "burgundy": "red", "maroon": "red",
    # oranges ↔ pinks
    "coral": "orange", "peach": "orange", "salmon": "orange",
    # greens
    "sage": "green", "olive": "green", "chartreuse": "green", "lime": "green",
    "mint": "green", "emerald": "green", "forest": "green",
    # blues
    "sky": "blue", "azure": "blue", "cerulean": "blue", "cobalt": "blue",
    "ultramarine": "blue", "navy": "blue", "cyan": "blue",
    # blue-greens
    "teal": "green", "turquoise": "green",
    # purples
    "violet": "purple", "indigo": "purple", "lavender": "purple",
    "lilac": "purple", "mauve": "purple", "magenta": "purple", "fuchsia": "purple",
    # pink family
    "rose": "pink", "blush": "pink", "hot_pink": "pink",
    # meta
    "pastel": "pastel", "monochrome": "monochrome"
}

# Art technique keywords for detection
TECHNIQUE_WORDS = [
    "chiaroscuro", "sfumato", "impasto", "glazing", "collage", "drypoint", "etching", "lithograph",
    "pointillism", "drip", "stain", "wash", "hatching", "cross-hatching", "gestural", "palette knife",
    "flat color", "hard edge", "geometric", "soft edge", "photographic", "grainy", "textured", "brushwork",
    "dramatic contrast", "thick application of paint"
]

# Art movement keywords for style detection
MOVEMENT_KEYS_HINTS = [
    "impressionism", "expressionism", "romanticism", "abstract expressionism", "surrealism", "baroque",
    "realism", "cubism", "fauvism", "minimalism", "symbolism", "pop art", "renaissance", "mannerism"
]

# Mood words derived from canonical emotions
MOOD_WORDS = sorted({w for vs in CANON_EMOTIONS.values() for w in vs})

def _lower(s: str) -> str:
    """Convert string to lowercase and strip whitespace"""
    return s.lower().strip()

def _tokenize(text: str) -> List[str]:
    """Extract words with letters and optional hyphens (e.g., blue-green)"""
    return re.findall(r"[A-Za-z][A-Za-z\-]+", (text or "").lower())

def _canon_emotions_from_text(text: str) -> List[str]:
    """Extract canonical emotions from text based on keyword matching"""
    toks = set(_tokenize(text))
    found = []
    for canon, variants in CANON_EMOTIONS.items():
        if any(v in toks for v in variants):
            found.append(canon)
    return sorted(set(found))

def normalize_color(token: str, color_emotions: Dict[str, Any]) -> str:
    """
    Map a detected color token to a key that exists in color_emotions.json.
    Preference order: 1) exact match, 2) alias mapping, 3) fallback to token
    """
    if token in color_emotions:
        return token
    if token in COLOR_ALIASES:
        base = COLOR_ALIASES[token]
        return base if base in color_emotions else token
    return token

# ---------- Data Loading ----------
def load_art_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load art movements and color emotions data from JSON files"""
    with open('data/art_movements.json', 'r', encoding='utf-8') as f:
        art_movements = json.load(f)
    with open('data/color_emotions.json', 'r', encoding='utf-8') as f:
        color_emotions = json.load(f)
    return art_movements, color_emotions

# ---------- Visual Analysis Parsing ----------
# Regex patterns for extracting sections from visual analysis
SECTION_PATTERNS = {
    "composition": r"2\)\s*Composition\s*&\s*technique(.*?)(?:\n\d\)|\Z)",
    "color_light": r"3\)\s*Color\s*&\s*light(.*?)(?:\n\d\)|\Z)",
    "mood": r"4\)\s*Mood\s*&\s*atmosphere(.*?)(?:\n\d\)|\Z)",
    "style": r"5\)\s*Possible\s*style/movement\s*\(.*?\)(.*?)(?:\n\d\)|\Z)",
}

def parse_visual_analysis(visual_text: str) -> Dict[str, Any]:
    """Extract structured information from guided vision output"""
    text = visual_text or ""
    out = {
        "techniques": [],
        "colors": [],
        "temperature": None,
        "contrast": None,
        "mood_words": [],
        "style_hypotheses": []
    }

    # Extract composition & technique information
    m = re.search(SECTION_PATTERNS["composition"], text, flags=re.IGNORECASE | re.DOTALL)
    comp = m.group(1) if m else ""
    comp_tokens = " ".join(_tokenize(comp))
    out["techniques"] = sorted({t for t in TECHNIQUE_WORDS if t in comp_tokens})

    # Extract color & light information
    m = re.search(SECTION_PATTERNS["color_light"], text, flags=re.IGNORECASE | re.DOTALL)
    col = m.group(1) if m else ""
    tokens = _tokenize(col)

    # Detect color words
    out["colors"] = sorted({c for c in COLOR_WORDS if c in tokens})

    # Simple heuristics for temperature/contrast
    out["temperature"] = "warm" if "warm" in tokens else ("cool" if "cool" in tokens else None)
    out["contrast"] = (
        "high" if ("high" in tokens and "contrast" in tokens)
        else ("low" if ("low" in tokens and "contrast" in tokens) else None)
    )

    # Extract mood & atmosphere
    m = re.search(SECTION_PATTERNS["mood"], text, flags=re.IGNORECASE | re.DOTALL)
    moodtxt = m.group(1) if m else ""
    mood_tokens = _tokenize(moodtxt)
    out["mood_words"] = sorted({w for w in MOOD_WORDS if w in mood_tokens})

    # Extract style/movement hypotheses
    m = re.search(SECTION_PATTERNS["style"], text, flags=re.IGNORECASE | re.DOTALL)
    sty = (m.group(1) if m else "").lower()
    for key in MOVEMENT_KEYS_HINTS:
        if key in sty:
            # Determine confidence level
            conf = "low"
            if "high" in sty and "confidence" in sty:
                conf = "high"
            elif "medium" in sty and "confidence" in sty:
                conf = "medium"
            out["style_hypotheses"].append({"movement_key": key, "confidence": conf})

    return out

# ---------- Emotion Analysis ----------
def analyze_emotion_text(text: str) -> List[Dict[str, Any]]:
    """Use transformer model to analyze emotional content of text"""
    try:
        return emotion_classifier(text)
    except Exception:
        return [{"label": "UNKNOWN", "score": 0.0}]

def extract_emotion_keywords(user_text_or_list) -> List[str]:
    """Extract emotion keywords from user input (text or list)"""
    if isinstance(user_text_or_list, list):
        # Handle dropdown selections like "Calm/Peaceful" -> "calm"
        raw = " ".join([x.split("/")[0] for x in user_text_or_list])
    else:
        raw = user_text_or_list or ""
    return _canon_emotions_from_text(raw)

# ---------- Scoring Functions ----------
def _overlap(a: List[str], b: List[str]) -> float:
    """Calculate Jaccard similarity between two lists"""
    A, B = set(a), set(b)
    return 0.0 if not A or not B else len(A & B) / len(A | B)

def _contains_any(text_list: List[str], ref_list: List[str]) -> float:
    """Check if any elements overlap between lists"""
    return 1.0 if set(text_list) & set(ref_list) else 0.0

def rank_art_movements(
    user_emotions: List[str],
    visual_cues: Dict[str, Any],
    art_movements: Dict[str, Any],
    color_emotions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Rank art movements based on composite scoring from emotions and visual analysis"""
    results = []

    # Derive color-based moods from detected colors
    detected_color_moods: List[str] = []
    for c in visual_cues.get("colors", []):
        base = normalize_color(c, color_emotions)
        if base in color_emotions:
            detected_color_moods += color_emotions[base]["emotions"]
    detected_color_moods = _canon_emotions_from_text(" ".join(detected_color_moods))

    # Create style hint map for bonus scoring
    style_hints = {h["movement_key"]: h.get("confidence", "low")
                   for h in visual_cues.get("style_hypotheses", [])}

    for key, mv in art_movements.items():
        mv_emotions = [_lower(e) for e in mv.get("emotions", [])]
        mv_techniques = [_lower(t) for t in mv.get("techniques", [])]
        mv_visual_cues = [_lower(v) for v in mv.get("visual_cues", [])]
        mv_palette_cues = [_lower(p) for p in mv.get("palette_cues", [])]

        # Calculate component scores
        emo_score = _overlap(user_emotions, mv_emotions)
        color_score = _overlap(detected_color_moods, mv_emotions) * 0.8
        tech_score = _overlap(visual_cues.get("techniques", []), mv_techniques)
        cue_score = _contains_any(visual_cues.get("mood_words", []), mv_visual_cues)

        # Style hint bonus
        style_bonus = 0.0
        if key in style_hints:
            style_bonus = {"low": 0.05, "medium": 0.12, "high": 0.2}.get(style_hints[key], 0.05)

        # Temperature/palette bonus
        temp_bonus = 0.05 if visual_cues.get("temperature") in mv_palette_cues else 0.0

        # Calculate final composite score
        score = (
            0.10 * emo_score +
            0.55 * tech_score +
            0.30 * color_score +
            0.05 * cue_score +
            style_bonus + temp_bonus
        )

        # Build reasons list for explanation
        reasons = []
        if emo_score > 0:
            reasons.append(f"Emotion overlap: {sorted(set(user_emotions) & set(mv_emotions))}")
        if tech_score > 0 and visual_cues.get("techniques"):
            reasons.append(f"Technique match: {sorted(set(visual_cues['techniques']) & set(mv_techniques))}")
        if color_score > 0 and detected_color_moods:
            reasons.append(f"Color-mood alignment: {detected_color_moods}")
        if cue_score > 0 and visual_cues.get("mood_words"):
            reasons.append(f"Mood cues: {sorted(set(visual_cues['mood_words']) & set(mv_visual_cues))}")
        if style_bonus > 0:
            reasons.append(f"Style hint in vision analysis (confidence {style_hints[key]})")
        if temp_bonus > 0:
            reasons.append("Palette temperature matches movement profile")

        results.append({
            "movement_key": key,
            "movement": mv,
            "relevance_score": round(float(score), 3),
            "overlap_emotions": sorted(set(user_emotions) & set(mv_emotions)),
            "reasons": reasons
        })

    # Sort by relevance score and return top 3
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:3]

# ---------- Public API ----------
def generate_art_context(user_emotion_text: str, visual_analysis: str) -> Dict[str, Any]:
    """
    Main function to generate art context based on user emotions and visual analysis
    """
    # Load data files
    art_movements, color_emotions = load_art_data()

    # Analyze emotions
    emotion_analysis = analyze_emotion_text(user_emotion_text)
    emotion_keywords = extract_emotion_keywords(user_emotion_text)

    # Parse visual analysis for structured cues
    visual_cues = parse_visual_analysis(visual_analysis)

    # Rank movements using composite scoring
    matches = rank_art_movements(
        user_emotions=emotion_keywords,
        visual_cues=visual_cues,
        art_movements=art_movements,
        color_emotions=color_emotions
    )

    return {
        "detected_emotions": emotion_analysis,
        "emotion_keywords": emotion_keywords,
        "visual_cues": visual_cues,
        "art_movement_matches": matches,
        "visual_analysis": visual_analysis
    }

def _call_llm(system: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API with fallback handling for different SDK versions"""
    client = OpenAI()

    try:
        # Try newer SDK: Responses API
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        return resp.output_text or ""

    except (openai.BadRequestError, openai.APIStatusError, AttributeError, TypeError):
        # Fallback to Chat Completions API
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return chat.choices[0].message.content or ""

def _shorten(obj: Any, max_chars: int = 1200) -> str:
    """Safely stringify and truncate long objects for compact prompts"""
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        s = s[:max_chars] + " …"
    return s

def generate_emotional_visual_analysis(
    user_emotions: str,
    visual_analysis: str,
    dominant_colors: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate comprehensive analysis of how visual elements create emotional responses
    
    Args:
        user_emotions: User's emotional response (text)
        visual_analysis: Detailed visual analysis of the artwork
        dominant_colors: Color data from color analysis
        model: OpenAI model to use
    
    Returns:
        Markdown formatted emotional-visual analysis
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Extract key color information for the prompt
    color_summary = []
    for color in dominant_colors[:5]:  # Top 5 colors
        color_summary.append(f"{color['name']} ({color['pct']:.1%})")
    color_text = ", ".join(color_summary)
    
    # Truncate visual analysis for prompt efficiency
    visual_text = _shorten(visual_analysis, max_chars=1500)
    
    system = (
        "You are an art psychologist and visual perception expert analyzing how specific "
        "visual elements in artworks create emotional responses in viewers. Provide clear, "
        "evidence-based analysis connecting concrete visual features to psychological impact. "
        "Focus on observable artistic elements and their documented emotional effects."
    )
    
    user_prompt = f"""
Analyze how the visual elements in this artwork create the emotional response described by the viewer.

**Viewer's Emotional Response:**
{user_emotions}

**Visual Analysis of the Artwork:**
{visual_text}

**Dominant Colors:**
{color_text}

Please provide a detailed analysis addressing these aspects:

**Color Psychology and Emotional Impact**
Explain how the specific colors present in this artwork (particularly the dominant ones) contribute to the viewer's emotional response. Reference established color psychology principles and how these colors work together to create mood.

**Compositional Elements and Emotional Flow**
Analyze how the composition, lines, shapes, and spatial arrangements guide the viewer's eye and create emotional responses. Discuss how elements like balance, symmetry, leading lines, and focal points affect the psychological experience.

**Light, Shadow, and Atmospheric Effects**
Examine how the treatment of light, shadow, contrast, and atmospheric qualities in the work contribute to the emotional mood. Consider how lighting creates drama, serenity, mystery, or other feelings.

**Texture, Brushwork, and Material Qualities**
Discuss how the visible artistic technique (brushwork, surface texture, medium handling) affects the emotional impact. Consider whether rough, smooth, gestural, or precise techniques contribute to the viewer's feelings.

**Scale, Proportion, and Visual Weight**
Analyze how the size relationships, proportions, and visual weight distribution in the composition affect the psychological impact and emotional response.

Conclude with a synthesis of how these visual elements work together to create the specific emotional response the viewer described. Be specific about which visual cues generate which emotional effects, and explain the psychological mechanisms involved.

Keep the analysis grounded in visual evidence and established principles of visual perception and psychology.
"""

    return _call_llm(system=system, user_prompt=user_prompt, model=model)

def generate_deeper_art_context(
    movement: Dict[str, Any],
    reasons: Optional[List[str]],
    user_emotions: Any,
    visual_cues: Any,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate professional, detailed art historical analysis using LLM
   
    Args:
        movement: Selected art movement dictionary from JSON
        reasons: Computed reasons for the match
        user_emotions: User's emotion input (text or list)
        visual_cues: Visual analysis results
        model: OpenAI model to use
   
    Returns:
        Markdown formatted art historical analysis
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not found. Set it in env or Streamlit secrets.")

    # Extract movement information
    movement_name = movement.get("name", "Unknown movement")
    movement_desc = movement.get("description", "")
    movement_artists = ", ".join(movement.get("key_artists", [])) or "—"
    movement_emolink = movement.get("emotional_connection", "")

    # Format reasons for analysis
    reasons_text = "• " + "\n• ".join((reasons or [])[:5]) if reasons else "—"

    # Keep visual cues compact for the prompt
    visual_text = _shorten(visual_cues, max_chars=1200)

    # Handle emotions as list or string
    if isinstance(user_emotions, (list, tuple, set)):
        emotions_text = ", ".join(user_emotions)
    else:
        emotions_text = str(user_emotions)

    # System prompt for professional, neutral tone
    system = (
        "You are an art historian and museum educator writing for a general audience. "
        "Provide clear, accessible analysis that is professional, neutral, and evidence-based. "
        "Do not invent facts beyond the provided context. Maintain an educational tone "
        "without being overly casual or personal. Focus on concrete visual elements and "
        "historical context rather than subjective interpretations."
    )

    # User prompt with structured sections
    user_prompt = f"""
Analyze this artwork within its art historical context based on the following information:

**User's Emotional Response:**
{emotions_text}

**Visual Analysis:**
{visual_text}

**Identified Art Movement:**
- Name: {movement_name}
- Key Artists: {movement_artists}
- Historical Context: {movement_desc}
- Emotional Characteristics: {movement_emolink}

**Match Reasoning:**
{reasons_text}

Please provide a structured analysis in Markdown format with these sections:

### Art Historical Context
Situate this work within the {movement_name} movement. Discuss the historical period, key characteristics of the movement, and how this artwork exemplifies those principles. Reference 1-2 specific artists from the movement ({movement_artists}) and explain their relevance to understanding this work. Include information about the social, cultural, or artistic conditions that gave rise to this movement.

### Technical and Stylistic Analysis  
Examine the artistic techniques, materials, and stylistic approaches visible in this work. Explain how these technical choices serve the artistic and emotional goals of the {movement_name} movement. Discuss any innovations or conventions that are particularly relevant. Connect the technical execution to the broader aesthetic goals of the movement.

### Further Exploration
Provide 3-4 specific suggestions for what viewers should look for when examining this artwork more closely. Focus on details that would help confirm or deepen understanding of the art historical connections identified. Include suggestions for related artworks, artists, or museum collections that would enhance understanding of this artistic tradition.

Keep the analysis professional, informative, and grounded in art historical scholarship. Avoid speculation beyond what can be supported by the visual evidence and movement characteristics provided. Write in clear, accessible language suitable for museum visitors or art students.
"""

    return _call_llm(system=system, user_prompt=user_prompt, model=model)