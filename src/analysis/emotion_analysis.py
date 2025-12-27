"""
Emotion Analysis Visualization Tool

This script visualizes emotion data from conversation history to help identify
trends and patterns in the user's emotional state over time.

Usage:
    python emotion_analysis.py --user <user_id> [--days <days>] [--output <image_path>]

Options:
    --user: User ID for conversation tracking
    --days: Number of days of history to analyze (default: 30)
    --output: Output path for the visualization image
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.conversation_tracker import ConversationTracker

# Define emotion categories
POSITIVE_EMOTIONS = ["joy", "happy", "excited", "content", "grateful", "satisfied", "optimistic"]
NEGATIVE_EMOTIONS = ["sad", "angry", "frustrated", "anxious", "fearful", "stressed", "depressed"]
NEUTRAL_EMOTIONS = ["neutral", "calm", "thoughtful", "contemplative", "reflective"]

def categorize_emotion(emotion: str) -> str:
    """Categorize an emotion as positive, negative, or neutral"""
    emotion = emotion.lower()
    if any(pos in emotion for pos in POSITIVE_EMOTIONS):
        return "positive"
    elif any(neg in emotion for neg in NEGATIVE_EMOTIONS):
        return "negative"
    else:
        return "neutral"

def extract_emotion_data(tracker: ConversationTracker, days: int = 30) -> pd.DataFrame:
    """
    Extract emotion data from conversation history
    
    Args:
        tracker: Conversation tracker instance
        days: Number of days of history to extract
        
    Returns:
        DataFrame with emotion data
    """
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()
    
    # Get all conversations in date range
    conversations = []
    
    # Get the conversation data from the tracker
    export_path = tracker.export_conversations(
        start_date=start_date_str,
        end_date=end_date_str
    )
    
    if not export_path or not os.path.exists(export_path):
        print("No conversation data found")
        return pd.DataFrame()
    
    # Load the exported data
    with open(export_path, 'r') as f:
        export_data = json.load(f)
        conversations = export_data.get("conversations", [])
    
    # Extract emotion data
    emotion_data = []
    
    for conv in conversations:
        try:
            # Get timestamp
            timestamp = datetime.fromisoformat(conv.get("timestamp", end_date.isoformat()))
            
            # Get primary emotion
            emotion = conv.get("primary_emotion")
            if not emotion:
                # Try to extract from metadata
                metadata = conv.get("metadata", {})
                if "emotion" in metadata:
                    emotion_info = metadata["emotion"]
                    if isinstance(emotion_info, dict):
                        emotion = emotion_info.get("primary_emotion")
            
            if emotion:
                # Add to the list - safely extract message with None handling
                user_message = conv.get("user_message") or ""
                message_snippet = str(user_message)[:50] if user_message else ""
                emotion_data.append({
                    "timestamp": timestamp,
                    "emotion": emotion,
                    "category": categorize_emotion(emotion),
                    "message": message_snippet  # Include snippet of message for context
                })
        except Exception as e:
            print(f"Error processing conversation: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(emotion_data)
    
    # Sort by timestamp
    if not df.empty:
        df = df.sort_values("timestamp")
    
    return df

def plot_emotion_timeline(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """
    Create visualization of emotions over time
    
    Args:
        df: DataFrame with emotion data
        output_path: Path to save the visualization
    """
    if df.empty:
        print("No emotion data to visualize")
        return
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Emotion Analysis Over Time", fontsize=16)
    
    # Plot 1: Emotion scatter plot
    categories = {
        "positive": {"color": "green", "marker": "o"},
        "negative": {"color": "red", "marker": "s"},
        "neutral": {"color": "blue", "marker": "^"}
    }
    
    for category, props in categories.items():
        category_data = df[df["category"] == category]
        if not category_data.empty:
            ax1.scatter(
                category_data["timestamp"], 
                category_data["emotion"],
                color=props["color"],
                marker=props["marker"],
                alpha=0.7,
                label=category
            )
    
    # Add legend
    ax1.legend()
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add labels
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Detected Emotion")
    ax1.set_title("Individual Emotions Detected in Conversations")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Emotional state distribution over time
    # Create daily aggregates
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby(["date", "category"]).size().unstack(fill_value=0)
    
    # Ensure all categories are present
    for category in ["positive", "negative", "neutral"]:
        if category not in daily_counts.columns:
            daily_counts[category] = 0
    
    # Calculate percentages
    daily_totals = daily_counts.sum(axis=1)
    daily_percentages = daily_counts.div(daily_totals, axis=0) * 100
    
    # Plot stacked area chart
    daily_percentages.plot.area(
        ax=ax2,
        stacked=True,
        color=["green", "red", "blue"],
        alpha=0.7
    )
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add labels
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Percentage")
    ax2.set_title("Emotional State Distribution Over Time")
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Emotion Analysis Visualization Tool")
    parser.add_argument("--user", default="default_user", help="User ID for conversation tracking")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to analyze")
    parser.add_argument("--output", help="Output path for the visualization image")
    
    args = parser.parse_args()
    
    # Initialize the conversation tracker
    tracker = ConversationTracker(user_id=args.user)
    print(f"Initialized conversation tracker for user: {args.user}")
    
    # Extract emotion data
    print(f"Analyzing emotion data for the past {args.days} days...")
    df = extract_emotion_data(tracker, days=args.days)
    
    if df.empty:
        print("No emotion data found for the specified time period")
        return
    
    print(f"Found {len(df)} conversations with emotion data")
    
    # Create visualization
    plot_emotion_timeline(df, args.output)

if __name__ == "__main__":
    asyncio.run(main())
