import streamlit as st
from PIL import Image
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

from vision_analysis import analyze_artwork_visual
from emotion_processing import generate_art_context
from emotion_processing import generate_deeper_art_context
from color_utils import extract_dominant_colors

st.set_page_config(page_title="Art For All", layout="wide")

# Set the style for matplotlib plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Clean headers without emojis
st.title("Art for All")
st.subheader("Discover the emotional language of art")

# Image upload section
uploaded_file = st.file_uploader(
    "Upload an artwork image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload any artwork image you'd like to explore"
)

# Initialize so we can reference them safely later
analyze_button = False
user_emotion = None
image = None

def create_color_visualizations(image):
    """
    Create comprehensive color analysis visualizations for the uploaded artwork

    Args:
        image: PIL Image object

    Returns:
        tuple: (dominant_colors_data, fig) for display
    """
    # Extract dominant colors using our color utility
    dominant_colors = extract_dominant_colors(image, k=8)

    # Convert image to numpy array for additional analysis
    img_array = np.array(image.convert('RGB'))

    # Flatten the image to get all pixel values
    pixels = img_array.reshape(-1, 3)

    # Create visualizations
    fig = plt.figure(figsize=(15, 12))

    # 1. Dominant Colors Pie Chart
    ax1 = plt.subplot(2, 3, 1)
    colors_for_pie = [color['hex'] for color in dominant_colors]
    sizes = [color['pct'] * 100 for color in dominant_colors]
    labels = [f"{color['name']}\n({color['pct']:.1%})" for color in dominant_colors]

    ax1.pie(
        sizes,
        labels=labels,
        colors=colors_for_pie,
        autopct='',
        startangle=90,
        textprops={'fontsize': 8}
    )
    ax1.set_title('Dominant Color Distribution', fontsize=12, fontweight='bold', pad=20)

    # 2. Color Palette Bar
    ax2 = plt.subplot(2, 3, 2)
    color_names = [color['name'] for color in dominant_colors]
    color_percentages = [color['pct'] * 100 for color in dominant_colors]
    bars = ax2.barh(
        range(len(color_names)),
        color_percentages,
        color=[color['hex'] for color in dominant_colors]
    )
    ax2.set_yticks(range(len(color_names)))
    ax2.set_yticklabels(color_names, fontsize=10)
    ax2.set_xlabel('Percentage (%)', fontsize=10)
    ax2.set_title('Color Prominence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, color_percentages)):
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%',
            va='center',
            fontsize=9
        )

    # 3. RGB Channel Distribution
    ax3 = plt.subplot(2, 3, 3)
    colors_rgb = ['red', 'green', 'blue']
    for i, color in enumerate(colors_rgb):
        ax3.hist(pixels[:, i], bins=50, alpha=0.7, label=f'{color.title()} Channel', color=color, density=True)
    ax3.set_xlabel('Intensity Value (0-255)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('RGB Channel Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Brightness Distribution
    ax4 = plt.subplot(2, 3, 4)
    # Convert to grayscale for brightness analysis
    brightness = np.mean(pixels, axis=1)
    ax4.hist(brightness, bins=50, color='gray', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Brightness Level (0-255)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Brightness Distribution', fontsize=12, fontweight='bold')
    ax4.axvline(np.mean(brightness), color='red', linestyle='--', label=f'Mean: {np.mean(brightness):.1f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Color Temperature Analysis
    ax5 = plt.subplot(2, 3, 5)
    # Calculate warm vs cool colors based on dominant colors
    warm_tokens = ['red', 'orange', 'yellow', 'pink', 'brown', 'coral', 'peach', 'salmon']
    cool_tokens = ['blue', 'green', 'purple', 'cyan', 'teal', 'turquoise', 'navy']

    warm_percentage = sum(
        [color['pct'] for color in dominant_colors if any(w in color['name'].lower() for w in warm_tokens)]
    )
    cool_percentage = sum(
        [color['pct'] for color in dominant_colors if any(c in color['name'].lower() for c in cool_tokens)]
    )
    neutral_percentage = max(0.0, 1 - warm_percentage - cool_percentage)

    temp_data = [warm_percentage * 100, cool_percentage * 100, neutral_percentage * 100]
    temp_labels = ['Warm', 'Cool', 'Neutral']
    temp_colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']

    bars = ax5.bar(temp_labels, temp_data, color=temp_colors, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Percentage (%)', fontsize=10)
    ax5.set_title('Color Temperature Analysis', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    for bar, pct in zip(bars, temp_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    # 6. Top Colors - Radial View (polar)
    angles = np.linspace(0, 2 * np.pi, len(dominant_colors[:5]), endpoint=False)
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    bars = ax6.bar(
        angles,
        [color['pct'] for color in dominant_colors[:5]],
        color=[color['hex'] for color in dominant_colors[:5]],
        alpha=0.8,
        width=0.8
    )
    ax6.set_theta_zero_location('N')
    ax6.set_theta_direction(-1)
    ax6.set_title('Top 5 Colors - Radial View', fontsize=12, fontweight='bold', pad=20)
    if dominant_colors:
        ax6.set_ylim(0, max([color['pct'] for color in dominant_colors[:5]]) * 1.1)

    for angle, color in zip(angles, dominant_colors[:5]):
        ax6.text(angle, color['pct'] * 1.2, color['name'], ha='center', va='center', fontsize=9)

    plt.tight_layout()

    return dominant_colors, fig

def display_color_statistics(dominant_colors):
    """Display color statistics in a formatted table"""

    # Create a dataframe for better display
    color_df = pd.DataFrame(
        [
            {
                'Color': color['name'].title(),
                'Hex Code': color['hex'],
                'Percentage': f"{color['pct']:.1%}",
                'RGB': f"({int(color['hex'][1:3], 16)}, {int(color['hex'][3:5], 16)}, {int(color['hex'][5:7], 16)})",
            }
            for color in dominant_colors
        ]
    )

    st.subheader("Color Analysis Summary")

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Colors Detected", len(dominant_colors))

    with col2:
        dominant_color = dominant_colors[0]
        st.metric("Most Dominant Color", dominant_color['name'].title(), f"{dominant_color['pct']:.1%}")

    with col3:
        # Calculate color diversity (inverse of how concentrated the top color is)
        diversity_score = 1 - dominant_colors[0]['pct']
        st.metric("Color Diversity", f"{diversity_score:.1%}")

    with col4:
        # Calculate if image is more warm or cool
        warm_keywords = ['red', 'orange', 'yellow', 'pink', 'brown']
        warm_pct = sum(
            [color['pct'] for color in dominant_colors if any(w in color['name'].lower() for w in warm_keywords)]
        )
        temp = "Warm" if warm_pct > 0.5 else ("Cool" if warm_pct < 0.3 else "Balanced")
        st.metric("Color Temperature", temp)

    st.subheader("Detailed Color Breakdown")

    # Style helper: color the "Hex Code" cell with its own color
    def color_row(row):
        hx = row.get("Hex Code", "#ffffff")
        # Choose text color by brightness threshold
        try:
            brightness_sum = sum(int(hx[i:i + 2], 16) for i in (1, 3, 5))
        except Exception:
            brightness_sum = 765  # default to dark text
        text_color = "white" if brightness_sum < 300 else "black"
        return [f'background-color: {hx}; color: {text_color}']

    # Apply styling ONLY to the "Hex Code" column (so the function has access to it)
    styled_df = color_df.style.apply(color_row, axis=1, subset=['Hex Code'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Your artwork", use_container_width=True)

    with col2:
        st.write("## How does this artwork make you feel?")

        # Emotion input options
        emotion_input_type = st.radio("Choose how to describe your emotions:", ["Free text", "Select from options"])

        if emotion_input_type == "Free text":
            user_emotion = st.text_area(
                "Describe your emotional response:", placeholder="I feel calm and peaceful when I look at this..."
            )
        else:
            emotion_options = [
                "Calm/Peaceful",
                "Energetic/Excited",
                "Melancholy/Sad",
                "Confused/Overwhelmed",
                "Joyful/Happy",
                "Anxious/Tense",
                "Contemplative/Thoughtful",
                "Angry/Frustrated",
            ]
            user_emotion = st.multiselect("Select emotions:", emotion_options)

        analyze_button = st.button("Analyze Art & Emotions", type="primary")

# Results section - only show if all conditions are met
if uploaded_file and analyze_button and user_emotion:
    st.write("---")
    st.write("# Analysis Results")

    with st.spinner("Analyzing your artwork and emotions..."):
        # Get enhanced visual analysis
        visual_analysis = analyze_artwork_visual(image)

        # Process emotions and match to art context
        emotion_text = user_emotion if isinstance(user_emotion, str) else " ".join(user_emotion)
        art_context = generate_art_context(user_emotion, visual_analysis)

        # Generate color visualizations
        dominant_colors, color_fig = create_color_visualizations(image)

    # Create main tabs for all analysis sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Formal Analysis", 
        "Color Data Analysis", 
        "Emotional Response Analysis", 
        "Art Historical Context"
    ])

    # Tab 1: Formal Analysis
    with tab1:
        st.write("### Professional Art Historical Examination")
        st.write(visual_analysis)

    # Tab 2: Color Data Analysis
    with tab2:
        st.write("### Comprehensive Visual Data Breakdown")

        # Display color statistics
        display_color_statistics(dominant_colors)

        # Display the comprehensive color analysis plots
        st.subheader("Visual Data Representations")
        st.pyplot(color_fig)

        # Additional insights based on color analysis
        st.subheader("Color Analysis Insights")

        # Generate insights based on the color data
        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.write("**Color Complexity:**")
            if len(dominant_colors) > 6:
                st.write("• This artwork shows high color complexity with multiple distinct hues")
            elif len(dominant_colors) > 3:
                st.write("• This artwork displays moderate color variety")
            else:
                st.write("• This artwork uses a limited, focused color palette")

            # Dominant color analysis
            primary_color = dominant_colors[0]
            if primary_color['pct'] > 0.4:
                st.write(f"• **{primary_color['name'].title()}** strongly dominates the composition ({primary_color['pct']:.1%})")
            else:
                st.write("• Colors are relatively well-distributed across the composition")

        with insights_col2:
            st.write("**Color Harmony:**")
            # Check for complementary colors
            color_names_lower = [color['name'].lower() for color in dominant_colors[:4]]
            if ('red' in color_names_lower and 'green' in color_names_lower) or \
               ('blue' in color_names_lower and 'orange' in color_names_lower) or \
               ('yellow' in color_names_lower and 'purple' in color_names_lower):
                st.write("• Contains complementary color relationships")

            # Check for analogous colors
            warm_count = sum(1 for name in color_names_lower if any(w in name for w in ['red', 'orange', 'yellow', 'pink']))
            cool_count = sum(1 for name in color_names_lower if any(c in name for c in ['blue', 'green', 'purple', 'cyan']))

            if warm_count >= 3:
                st.write("• Primarily warm color harmony")
            elif cool_count >= 3:
                st.write("• Primarily cool color harmony")
            else:
                st.write("• Mixed warm and cool color palette")

    # Tab 3: Emotional Response Analysis
    with tab3:
        st.write("### In-Depth Emotional and Psychological Impact")

        # Display user emotions clearly
        if isinstance(user_emotion, str):
            st.write(f"**Your emotional response:** {user_emotion}")
        else:
            st.write(f"**Selected emotions:** {', '.join(user_emotion)}")

        # Enhanced emotional analysis section
        st.subheader("Visual Elements and Their Emotional Impact")
        
        # Analyze how visual elements create emotional responses
        with st.spinner("Analyzing visual-emotional connections..."):
            # Get deeper emotional analysis using LLM
            if art_context.get('art_movement_matches'):
                best_match = art_context['art_movement_matches'][0]
                movement = best_match['movement']
                
                # Generate comprehensive emotional analysis
                from emotion_processing import generate_emotional_visual_analysis
                emotional_visual_analysis = generate_emotional_visual_analysis(
                    user_emotions=emotion_text,
                    visual_analysis=visual_analysis,
                    dominant_colors=dominant_colors,
                    model="gpt-4o-mini"
                )
                
                if emotional_visual_analysis.strip():
                    st.markdown(emotional_visual_analysis)
                else:
                    # Fallback analysis if LLM fails
                    st.write("**Color and Emotion Connection:**")
                    if art_context.get('emotion_keywords'):
                        emotion_keywords = art_context['emotion_keywords']
                        st.write(f"The emotional themes you identified ({', '.join(emotion_keywords)}) are supported by the artwork's visual characteristics.")
                        
                        # Connect colors to emotions
                        primary_color = dominant_colors[0]['name'].lower()
                        color_emotion_map = {
                            'red': 'intensity, passion, and energy',
                            'blue': 'calm, contemplation, and serenity',
                            'green': 'balance, nature, and tranquility',
                            'yellow': 'joy, optimism, and vitality',
                            'orange': 'warmth, friendliness, and energy',
                            'purple': 'mystery, spirituality, and depth',
                            'brown': 'earthiness, stability, and grounding',
                            'black': 'power, elegance, and mystery',
                            'white': 'purity, simplicity, and peace',
                            'gray': 'neutrality, balance, and sophistication'
                        }
                        
                        if any(key in primary_color for key in color_emotion_map.keys()):
                            for color_key, emotions in color_emotion_map.items():
                                if color_key in primary_color:
                                    st.write(f"The dominant {primary_color} tones typically evoke feelings of {emotions}, which aligns with your emotional response.")
                                    break

        # Show detected emotion keywords if available
        if art_context.get('emotion_keywords'):
            st.subheader("Identified Emotional Themes")
            st.write(f"**Key emotional patterns:** {', '.join(art_context['emotion_keywords'])}")

            # Add visual representation of emotions
            emotion_counts = Counter(art_context['emotion_keywords'])
            if len(emotion_counts) > 1:
                st.subheader("Emotional Theme Distribution")

                # Create a simple bar chart for emotions
                fig_emotion, ax = plt.subplots(1, 1, figsize=(10, 4))
                emotions = list(emotion_counts.keys())
                counts = list(emotion_counts.values())

                ax.bar(emotions, counts, color='lightblue', alpha=0.7, edgecolor='navy')
                ax.set_xlabel('Emotional Themes', fontsize=12)
                ax.set_ylabel('Prominence', fontsize=12)
                ax.set_title('Identified Emotional Themes in Your Response', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                if len(emotions) > 3:
                    plt.xticks(rotation=45, ha='right')

                plt.tight_layout()
                st.pyplot(fig_emotion)

    # Tab 4: Art Historical Context
    with tab4:
        st.write("### Comprehensive Scholarly Analysis")

        # Enhanced art historical analysis
        if art_context.get('art_movement_matches'):
            best_match = art_context['art_movement_matches'][0]
            movement = best_match['movement']
            movement_name = movement.get('name', 'Unknown movement')

            # Display basic movement information
            st.subheader(f"Identified Art Movement: {movement_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Key Artists:** {', '.join(movement.get('key_artists', []))}")
                st.write(f"**Historical Period:** {movement.get('description', 'Information not available')}")
                
            with col2:
                st.write(f"**Match Confidence:** {best_match.get('relevance_score', 0):.2%}")
                if best_match.get('reasons'):
                    st.write("**Matching Criteria:**")
                    for reason in best_match['reasons'][:3]:  # Show top 3 reasons
                        st.write(f"• {reason}")

            with st.spinner("Generating comprehensive art historical analysis..."):
                # Generate the detailed context using LLM
                llm_analysis = generate_deeper_art_context(
                    movement=movement,
                    reasons=best_match.get('reasons', []),
                    user_emotions=emotion_text,
                    visual_cues=visual_analysis,
                    model="gpt-4o-mini",
                )

                if llm_analysis.strip():
                    st.markdown(llm_analysis)
                else:
                    st.info("Unable to generate detailed historical analysis at this time. Please try again.")
        else:
            st.write("No specific art movement connections could be identified for this artwork and emotional response combination.")
            st.write("This may indicate a unique contemporary work or a piece that blends multiple artistic traditions.")

# Error handling wrapper
def safe_analyze():
    """Wrapper function for safe error handling during analysis"""
    try:
        if uploaded_file and analyze_button and user_emotion:
            # Analysis code is handled above
            pass
    except Exception:
        st.error("Something went wrong with the analysis. Please try again.")
        if st.checkbox("Show technical details"):
            st.code(traceback.format_exc())

# Clean sidebar with instructions
with st.sidebar:
    st.write("## How to use")
    st.write("1. Upload an artwork image")
    st.write("2. Describe how it makes you feel")
    st.write("3. Click analyze to discover art history connections")
    st.write("4. Explore results in different tabs")
    st.write("---")
    st.write("## Example artworks to try")
    st.write("- Monet's Water Lilies")
    st.write("- Van Gogh's Starry Night")
    st.write("- Rothko's color fields")
    st.write("- Renaissance portraits")
    st.write("- Abstract expressionist works")
    st.write("---")
    st.write("## Analysis Features")
    st.write("- **Formal Analysis**: Professional art historical examination")
    st.write("- **Color Data Analysis**: Statistical breakdown of artwork colors")
    st.write("- **Emotional Response**: In-depth psychological impact analysis")
    st.write("- **Art Historical Context**: Comprehensive scholarly analysis")