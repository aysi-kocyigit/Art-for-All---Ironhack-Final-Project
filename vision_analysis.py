import openai
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for API transmission.

    Args:
        image: PIL Image object

    Returns:
        str: Base64 encoded image string
    """
    # Ensure a JPEG-compatible mode
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def analyze_artwork_visual(image: Image.Image) -> str:
    """
    Analyze artwork using OpenAI Vision API with structured formal analysis approach.

    Args:
        image: PIL Image object of the artwork

    Returns:
        str: Detailed structured visual analysis of the artwork
    """
    try:
        base64_image = encode_image_to_base64(image)

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a professional art historian conducting a formal analysis of this artwork. "
                                "Provide a systematic, detailed examination following this exact structure. "
                                "Use clear headings and professional terminology throughout.\n\n"
                                
                                "**FORMAL ANALYSIS STRUCTURE:**\n\n"
                                
                                "### Scene & Subject Matter\n"
                                "Describe precisely what is depicted in the artwork. Include:\n"
                                "- Main subjects (figures, objects, landscapes, abstract elements)\n"
                                "- Their poses, actions, expressions, and interactions\n"
                                "- Setting and spatial context (interior, exterior, ambiguous space)\n"
                                "- Time indicators (time of day, season, historical period suggested)\n"
                                "- Narrative elements or symbolic content\n"
                                "- Any text, inscriptions, or readable elements\n"
                                "- Props, clothing, architectural details\n\n"
                                
                                "### Composition & Structure\n"
                                "Analyze the formal organization of the artwork:\n"
                                "- Overall compositional scheme (triangular, circular, linear, grid-based)\n"
                                "- Focal points and visual hierarchy\n"
                                "- Balance (symmetrical, asymmetrical, radial)\n"
                                "- Use of negative space and spatial relationships\n"
                                "- Leading lines and visual pathways\n"
                                "- Cropping decisions and format (vertical, horizontal, square)\n"
                                "- Viewpoint and perspective (eye level, bird's eye, worm's eye)\n"
                                "- Scale relationships between elements\n\n"
                                
                                "### Technique & Medium\n"
                                "Examine the technical execution and materials:\n"
                                "- Medium identification (oil, watercolor, acrylic, digital, mixed media, etc.)\n"
                                "- Brushwork characteristics (visible strokes, smooth blending, textural marks)\n"
                                "- Surface treatment (impasto, glazes, washes, dry brush)\n"
                                "- Mark-making qualities and artistic processes\n"
                                "- Level of finish (highly rendered vs. sketchy)\n"
                                "- Support material visible (canvas texture, paper grain)\n"
                                "- Any evidence of printmaking, photography, or digital processes\n\n"
                                
                                "### Color & Light Analysis\n"
                                "Provide comprehensive color and lighting analysis:\n"
                                "- Dominant color palette and color relationships\n"
                                "- Color harmony (complementary, analogous, triadic, monochromatic)\n"
                                "- Saturation levels (bright/vivid vs. muted/subdued)\n"
                                "- Value range (light to dark contrast)\n"
                                "- Color temperature (warm vs. cool dominance)\n"
                                "- Light source(s) and direction\n"
                                "- Shadow patterns and modeling of form\n"
                                "- Highlights and reflected light\n"
                                "- Atmospheric effects or color modulation\n\n"
                                
                                "### Mood & Atmospheric Qualities\n"
                                "Analyze the emotional and atmospheric impact:\n"
                                "- Overall mood or emotional tone conveyed\n"
                                "- How formal elements contribute to mood\n"
                                "- Atmospheric qualities (clarity, haziness, drama, serenity)\n"
                                "- Psychological space and viewer engagement\n"
                                "- Sense of movement or stillness\n"
                                "- Textural qualities that affect mood\n"
                                "- How color and light create emotional resonance\n"
                                "- Cultural or symbolic associations suggested by visual elements\n\n"
                                
                                "**ANALYSIS GUIDELINES:**\n"
                                "- Base all observations on visual evidence in the image\n"
                                "- Use precise art historical terminology\n"
                                "- Be specific about locations within the composition\n"
                                "- Quantify when possible (proportions, dominant elements)\n"
                                "- Focus on objective formal analysis\n"
                                "- Avoid speculation about artist identity or exact dating\n"
                                "- Write in professional, scholarly language\n"
                                "- Provide enough detail for comprehensive understanding\n\n"
                                
                                "Please conduct this formal analysis thoroughly, addressing each section "
                                "with specific visual observations that would help someone understand "
                                "the artwork's formal characteristics and visual impact."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        # Surface the error as text so the Streamlit app can display it
        return f"Error during visual analysis: {e}"