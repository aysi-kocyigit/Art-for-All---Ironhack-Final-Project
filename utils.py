import streamlit as st
import json
from typing import Dict, List
import re  

@st.cache_data
def load_all_art_data():
    """Load and cache all art-related data"""
    try:
        with open('data/art_movements.json', 'r') as f:
            movements = json.load(f)
        with open('data/color_emotions.json', 'r') as f:
            colors = json.load(f)
        with open('data/art_techniques.json', 'r') as f:
            techniques = json.load(f)
        
        return movements, colors, techniques
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return {}, {}, {}

def format_art_education_response(art_context: Dict) -> str:
    """Format the educational response in a friendly way"""
    response_parts = []
    
    if art_context['art_movement_matches']:
        movement = art_context['art_movement_matches'][0]['movement']
        overlap_emotions = art_context['art_movement_matches'][0]['overlap_emotions']
        
        response_parts.append(f"🎨 **Your emotional connection to art:**")
        response_parts.append(f"You felt {', '.join(overlap_emotions)} - this connects beautifully to **{movement['name']}**!")
        response_parts.append(f"\n{movement['emotional_connection']}")
        response_parts.append(f"\n**Learn more:** {movement['description']}")
        response_parts.append(f"\n**Artists to explore:** {', '.join(movement['key_artists'])}")
    
    return "\n".join(response_parts)

def prettify_visual_markdown(text: str) -> str:
    """Replace the model's raw headings with nicer markdown headings."""
    if not text:
        return ""
    replacements = [
        (r"(?im)^\s*1\)\s*SCENE\s*$", "### 🖼️ Scene & Setting"),
        (r"(?im)^\s*2\)\s*COMPOSITION\s*&\s*TECHNIQUE\s*$", "### 🧭 Composition & Technique"),
        (r"(?im)^\s*3\)\s*COLOR\s*&\s*LIGHT\s*$", "### 🎨 Color & Light"),
        (r"(?im)^\s*4\)\s*MOOD\s*&\s*ATMOSPHERE\s*$", "### 🌬️ Mood & Atmosphere"),
        (r"(?im)^\s*5\)\s*POSSIBLE\s*STYLE/MOVEMENT.*$", "### 🏛️ Possible Style/Movement"),
    ]
    out = text
    for pat, repl in replacements:
        out = re.sub(pat, repl, out)
    return out

def chips(words):
    """Render small 'chips' for keywords using markdown code spans."""
    if not words:
        return "—"
    return " ".join(f"`{w}`" for w in words)