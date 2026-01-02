# Personality Assessment Data

This directory contains data files for personality assessments used in the mental health chatbot.

## Files

- `big_five_questions.json`: Questions for the Big Five (OCEAN) personality assessment
- `mbti_questions.json`: Questions for the Myers-Briggs Type Indicator (MBTI) assessment
- `mbti_descriptions.json`: Descriptions of the 16 MBTI personality types

## Data Format

### Big Five Questions

```json
[
  {
    "id": 1,
    "text": "I am the life of the party.",
    "trait": "extraversion",
    "reversed": false
  },
  ...
]
```

### MBTI Questions

```json
[
  {
    "id": 1,
    "text": "At a party, you:",
    "options": [
      {"key": "A", "text": "Interact with many, including strangers", "dimension": "E"},
      {"key": "B", "text": "Interact with a few, known to you", "dimension": "I"}
    ]
  },
  ...
]
```

### MBTI Descriptions

```json
{
  "ISTJ": {
    "name": "The Inspector",
    "description": "Practical, fact-minded, reliable, and responsible...",
    "strengths": ["Organized", "Honest and direct", "Dedicated", "Strong-willed", "Responsible"],
    "weaknesses": ["Stubborn", "Insensitive", "Always by the book", "Judgmental", "Often unreasonably blame themselves"]
  },
  ...
}
```

## Usage

These data files are automatically loaded by the personality assessment modules. The files will be created with default data if they don't exist.
