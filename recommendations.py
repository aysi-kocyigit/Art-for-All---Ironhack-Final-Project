from typing import List, Dict

def get_similar_artworks(art_movement: str) -> List[Dict]:
    """Suggest similar artworks based on the matched movement"""
    # This could connect to museum APIs or be a curated list
    recommendations = {
        "impressionism": [
            {
                "title": "Water Lilies",
                "artist": "Claude Monet",
                "reason": "Similar peaceful, contemplative mood"
            }
        ]
        # Add more...
    }
    
    return recommendations.get(art_movement.lower(), [])
