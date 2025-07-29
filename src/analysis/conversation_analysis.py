"""
Conversation Analysis Utility

This script demonstrates how to use the ConversationTracker to access, analyze,
and export conversation history.

Usage:
    python conversation_analysis.py --user <user_id> [--action <action>] [--output <file_path>]

Actions:
    recent: Show recent conversations (default)
    emotion: Show emotion distribution
    search: Search conversations by query
    export: Export conversations to JSON
    cleanup: Remove old conversations
    stats: Show conversation statistics
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.conversation_tracker import ConversationTracker
from src.utils.helpers import format_timestamp

def print_conversation(conversation: Dict[str, Any], include_metadata: bool = False) -> None:
    """Print a conversation in a readable format"""
    print("\n" + "="*80)
    
    # Print timestamp
    timestamp = conversation.get("timestamp", "Unknown time")
    try:
        dt = datetime.fromisoformat(timestamp)
        formatted_time = dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        formatted_time = timestamp
    print(f"📅 {formatted_time}")
    
    # Print emotion if available
    emotion = conversation.get("primary_emotion")
    if emotion:
        print(f"😊 Emotion: {emotion}")
    
    # Print the conversation
    print(f"\n👤 User: {conversation.get('user_message', '')}")
    print(f"\n🤖 Assistant: {conversation.get('assistant_response', '')}")
    
    # Print metadata if requested
    if include_metadata and "metadata" in conversation:
        print("\n📊 Metadata:")
        for key, value in conversation.get("metadata", {}).items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    print("="*80)

def print_emotion_distribution(distribution: Dict[str, int]) -> None:
    """Print emotion distribution in a readable format"""
    print("\n📊 Emotion Distribution:")
    print("="*50)
    
    # Sort emotions by count
    sorted_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    total = sum(distribution.values())
    
    # Calculate max bar length
    max_bar_length = 40
    max_count = max(distribution.values()) if distribution else 0
    
    for emotion, count in sorted_emotions:
        percentage = (count / total) * 100 if total > 0 else 0
        bar_length = int((count / max_count) * max_bar_length) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"{emotion.ljust(15)} {str(count).rjust(5)} ({percentage:.1f}%) {bar}")
    
    print("="*50)
    print(f"Total conversations: {total}")

async def show_recent_conversations(tracker: ConversationTracker, limit: int = 5) -> None:
    """Show recent conversations"""
    print(f"\n📜 Showing {limit} most recent conversations")
    
    conversations = tracker.get_recent_conversations(limit=limit)
    
    if not conversations:
        print("No conversations found")
        return
    
    for conversation in conversations:
        print_conversation(conversation)

async def show_emotion_distribution(tracker: ConversationTracker, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> None:
    """Show emotion distribution"""
    print("\n📊 Emotion Distribution")
    if start_date or end_date:
        date_range = ""
        if start_date:
            date_range += f"from {start_date} "
        if end_date:
            date_range += f"to {end_date} "
        print(f"Date range: {date_range}")
    
    distribution = tracker.get_emotion_distribution(
        start_date=start_date,
        end_date=end_date
    )
    
    print_emotion_distribution(distribution)

async def search_conversations(tracker: ConversationTracker, 
                             query: str,
                             limit: int = 5) -> None:
    """Search conversations by query"""
    print(f"\n🔍 Searching conversations for: '{query}'")
    
    conversations = tracker.search_conversations(
        query=query,
        limit=limit
    )
    
    if not conversations:
        print("No matching conversations found")
        return
    
    print(f"Found {len(conversations)} matching conversations:")
    for conversation in conversations:
        print_conversation(conversation)

async def export_conversations(tracker: ConversationTracker,
                            output_path: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> None:
    """Export conversations to JSON"""
    print("\n📤 Exporting conversations")
    
    if start_date or end_date:
        date_range = ""
        if start_date:
            date_range += f"from {start_date} "
        if end_date:
            date_range += f"to {end_date} "
        print(f"Date range: {date_range}")
    
    output_file = tracker.export_conversations(
        output_path=output_path,
        start_date=start_date,
        end_date=end_date
    )
    
    if output_file:
        print(f"Conversations exported to: {output_file}")
    else:
        print("Failed to export conversations")

async def cleanup_old_conversations(tracker: ConversationTracker) -> None:
    """Remove old conversations"""
    print("\n🧹 Cleaning up old conversations")
    
    removed = tracker.cleanup_old_conversations()
    
    print(f"Removed {removed} old conversations")

async def show_statistics(tracker: ConversationTracker) -> None:
    """Show conversation statistics"""
    print("\n📊 Conversation Statistics")
    print("="*50)
    
    stats = tracker.get_statistics()
    
    print(f"Total conversations: {stats.get('total_conversations', 0)}")
    print(f"Total messages: {stats.get('total_messages', 0)}")
    
    if stats.get('first_conversation_date'):
        print(f"First conversation: {stats.get('first_conversation_date')}")
    if stats.get('last_conversation_date'):
        print(f"Last conversation: {stats.get('last_conversation_date')}")
    
    print("\nTop emotions:")
    if stats.get('top_emotions'):
        for emotion, count in stats.get('top_emotions', []):
            print(f"  {emotion}: {count}")
    else:
        print("  No emotion data available")
    
    print("="*50)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Conversation Analysis Utility")
    parser.add_argument("--user", default="default_user", help="User ID for conversation tracking")
    parser.add_argument("--action", choices=["recent", "emotion", "search", "export", "cleanup", "stats"], 
                      default="recent", help="Action to perform")
    parser.add_argument("--limit", type=int, default=5, help="Limit for conversation results")
    parser.add_argument("--query", help="Search query for the search action")
    parser.add_argument("--output", help="Output file path for exports")
    parser.add_argument("--start-date", help="Start date for filtering (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for filtering (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Initialize the conversation tracker
    tracker = ConversationTracker(user_id=args.user)
    print(f"Initialized conversation tracker for user: {args.user}")
    
    # Perform the requested action
    if args.action == "recent":
        await show_recent_conversations(tracker, args.limit)
    
    elif args.action == "emotion":
        await show_emotion_distribution(tracker, args.start_date, args.end_date)
    
    elif args.action == "search":
        if not args.query:
            print("Error: Search action requires a --query parameter")
            return
        await search_conversations(tracker, args.query, args.limit)
    
    elif args.action == "export":
        await export_conversations(tracker, args.output, args.start_date, args.end_date)
    
    elif args.action == "cleanup":
        await cleanup_old_conversations(tracker)
    
    elif args.action == "stats":
        await show_statistics(tracker)

if __name__ == "__main__":
    asyncio.run(main())
