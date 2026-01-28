import os
import time
import uuid
import json
import numpy as np
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from utils import (MEMREADER_PROMPT, 
                   get_embedding, parse_messages, LME_JUDGE_MODEL_TEMPLATE, 
                   LME_ANSWER_PROMPT, remove_code_blocks, extract_json)
from lme_eval import lme_grader
from datetime import datetime, timezone
import pytz
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from vector_db import VectorDBConfig, VectorDBFactory
# ==========================================
# 0. Setup & Prompts
# ==========================================
load_dotenv()

# ⚠️ 请确保环境变量中有 OPENAI_API_KEY 和 MILVUS_URI
# 如果是本地测试，确保 Docker 中 Milvus 已启动

llm_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Memory type specific prompts
EPISODIC_MEMORY_PROMPT = """You are a specialized Episodic Memory Manager.
Your role is to maintain and update episodic memories that capture specific events and interactions.

[INPUTS]
You will receive:
1. "New Episodic Memories": A list of episodic memories extracted from the latest conversation turn.
2. "Retrieved Episodic Memories": A list of retrieved episodic memories.

[OPERATIONS & GUIDELINES]
Compare New Episodic Memories with Retrieved Episodic Memories and perform the following operations:

1. **ADD MEMORY (create_episodic_memory)**
   - **Condition**: If a fact contains completely NEW episodic information not present in Existing Episodic Memories.
   - **Action**: Call `create_episodic_memory` with a concise summary of the episodic event.

2. **TRAJECTORIZE MEMORY (trajectorize_episodic_memory)**
   - **Condition**: If new and old episodic memories have similar content or are related events.
   - **Action**: Combine multiple related events into a timeline with a timestamp range, drawing conclusions from patterns (old versions remain for history).
   - **Constraint**: Provide the list of original contents as `source_old_contents`.
"""

SEMANTIC_MEMORY_PROMPT = """You are a specialized Semantic Memory Manager.
Your role is to maintain and update semantic memories that capture conceptual knowledge about people, places, objects, and concepts in the user's life.

[INPUTS]
You will receive:
1. "New Episodic Memories": New episodic memories extracted from the latest conversation.
2. "New Semantic Memories": New semantic memories extracted from the latest conversation.
3. "Retrieved Semantic Memories": A list of retrieved semantic memories.

[OPERATIONS & GUIDELINES]
Compare New Semantic Memories with Retrieved Semantic Memories (using Episodic Memories for context) and perform the following operations:

1. **ADD MEMORY (create_semantic_memory)**
   - **Condition**: If a fact contains completely NEW semantic information about the user's life not present in Existing Semantic Memories.
   - **Constraint**: Explicitly skip common knowledge already captured in Existing Semantic Memories.
   - **Action**: Call `create_semantic_memory` with a concise summary of the semantic information about the user's life.

2. **UPDATE MEMORY (update_semantic_memory)**
   - **Condition**: If a fact adds detail, corrects, or updates a specific Existing Semantic Memory about the user's life.
   - **Constraint**: You MUST use the original content of the memory as the `old_content` parameter.
   - **Logic**: Merge the old content and new fact into a comprehensive statement.

3. **DELETE MEMORY (delete_semantic_memory)**
   - **Condition**: If a fact explicitly contradicts an Existing Semantic Memory (and the new fact is trusted), or if the memory is no longer valid.
   - **Constraint**: You MUST use the original content of the memory as the `old_content` parameter.

4. **INFER MEMORY (infer_semantic_memory)**
   - **Condition**: Look for higher-level insights from multiple semantic memories about the user's life.
   - **Action**: Call `infer_semantic_memory` with a list of original contents of semantic memories acting as premises.

5. **NOOP (no_operation)**
   - **Condition**: If the fact is redundant (already exactly covered by memory), trivial, or represents common knowledge already captured in the model's parameters.
"""

CORE_MEMORY_PROMPT = """You are the Core Memory Manager.
Core memory is defined as persistent user information, including identity, preferences, personality traits, and key relationships. Your role is to maintain a comprehensive and cohesive **User Profile** (Core Memory) based on episodic and semantic memories.

[INPUTS]
You will receive:
1. "Current Core Memory": The existing structured user profile.
2. "New Episodic Memories": Recent events and interactions (as context).
3. "New Semantic Memories": New facts and concepts (as context).

[GOAL & STRUCTURE]
Create a structured, human-readable profile that captures the essence of the user. This is NOT a log of events, but a synthesized personality and identity guide.
You MUST organize the Core Memory into the following sections (use Markdown headers):
1.  **# Basic Information**: Name, age, occupation, location, key life history.
2.  **# Personality & Values**: Traits, communication style, core values, beliefs, behavior patterns.
3.  **# Interests & Preferences**: Hobbies, likes/dislikes, favorites (food, media, etc.), lifestyle choices.
4.  **# Relationships**: Key people in their life and the nature of the relationship.
5.  **# Current Focus & Goals**: Ongoing projects, immediate aspirations, current struggles.

[OPERATIONS & GUIDELINES]
Analyze the New Memories and update the Core Memory using the following operations:

1. **core_memory_append**
   - **Condition**: When adding a completely new section or distinct information that doesn't fit into existing narratives (Rarely used).
   - **Action**: Call `core_memory_append` to add new content to the end.
   
2. **core_memory_replace**
   - **Condition**: When a specific detail changes (e.g., job title, location) without affecting the overall structure.
   - **Action**: Call `core_memory_replace` to update a specific section or paragraph.

3. **core_memory_rewrite**
   - **Condition**: When new information requires a major update or reorganization to maintain flow and consistency.
   - **Action**: Call `core_memory_rewrite` with the complete rewritten profile.
   - **Guideline**: Synthesize new facts into the existing narrative. Don't just list them. Resolve conflicts by favoring new information.

[EXAMPLES]
# Basic Information
Sophia Lee is a ceramic artist based in San Francisco. She holds a degree in Art and Ceramics from SFSU.

# Personality & Values
Introverted and analytical, she values deep conversations over small talk. She emphasizes sustainability and non-toxic living.

# Interests & Preferences
- **Hobbies**: Playing guitar (Fender Stratocaster) and planning to learn ukulele.
- **Gaming**: Loves RPGs like Cyberpunk 2077.
- **Reading**: Enjoys poetry anthologies focusing on marginalized voices.

# Current Focus & Goals
Currently working on a personal knowledge management system and planning a project in Nigeria to connect villages to running water.
"""



# --- MEMREADER PROMPT FOR EPISODIC MEMORY --- 
MEMREADER_PROMPT_EPISODIC = """You are a dialogue memory generator. Your task is to write fragment episodic memories that capture only the NEW facts from the "Current Conversation" (do not repeat anything already covered in "Historical Memories").

Core principle:
Convert dialogue from first-person to third-person narration, preserving as much substantive information content from the original as possible, excluding only confirmed non-informative words.

What to preserve:
- All substantive information: people, events, times, places, causes, results, numbers, specific descriptions
- Original wording: Keep specific terms used in dialogue for titles, item names, activity descriptions, etc, numbers use Arabic numerals
- Emotional expressions: Retain explicit emotions and attitudes from original (like "happy", "worried", "likes"), but avoid adding subjective inferences not present in original

What to exclude:
- Only exclude purely functional words: greetings ("hi""bye"), confirmation words ("uh-huh""okay""yes"), meaningless fillers ("um""you know""like")

Time normalization:
- Preserve the original relative time expressions exactly as written (e.g., "last night", "this morning", "last Friday"). DO NOT convert relative time to absolute dates.

Style:
- Use English third-person narration.
- Write plain sentences (no lists/numbering/Markdown). Aim for 2-4 sentences, but allow longer to retain essential details.
- Use exact proper nouns as in the dialogue; do not replace/expand/infer names, organizations, or locations.
- Each memory should focus on one core fact or closely related fact group; avoid packing too many unrelated details into a single entry.

Episodic Memory Requirements:
- Focus on specific events, interactions, and experiences that happened at a particular time
- Capture concrete actions, conversations, and occurrences
- Include specific details about who did what, when, where, and why
- Each episodic memory should focus on one core event or closely related event group

Input:
Historical memories (do not repeat): {previous_summary}
Conversation date: {conversation_date}
Current conversation: {new_dialogue}

Please generate episodic memories from the conversation. If the current conversation has no substantial new content, provide a minimal 1-2 sentence summary of the core topic or attitude expressed in this turn (do NOT output "no significant additions" or similar empty statements).

**Episodic Memory Requirements:**
Each episodic memory entry MUST include:
1. **Timestamp**: A date in YYYY-MM-DD format
2. **Summary**: A brief summary of the event
3. **Detailed Description**: A comprehensive description capturing who, what, when, where, and why

**Example of CORRECT output:**
{{
  "episodic_memories": [
    "2024-03-15: Started new job at startup | Details: First day at TechCorp as senior engineer, met team lead Sarah, received onboarding materials, and set up work station.",
    "2024-03-16: Attended team meeting | Details: Participated in first team meeting, discussed project roadmap, learned about team's current priorities, and introduced self to colleagues."
  ]
}}

**Few Shot Example 1:**
Input:
**Today's Date**: 2025-11-15
**Previous Chat History**:
user: I'm looking for a gift for my wife.
assistant: Does she have any specific hobbies?
user: She loves outdoor sports.
assistant: How about tennis gear?

**Current Conversation Turn**:
user: She already has plenty of that. She is really into rock climbing these days.
assistant: Got it. Since she likes rock climbing, I recommend checking out the new indoor climbing gym downtown.
Output:
{{
  "episodic_memories": [
    "2025-11-15: Wife is into rock climbing | Details: The user mentioned that his wife is really into rock climbing these days and already has plenty of tennis gear. She loves outdoor sports.",
    "2025-11-15: Assistant recommends indoor climbing gym | Details: The assistant recommended checking out the new indoor climbing gym downtown since the user's wife likes rock climbing."
  ]
}}

**Few Shot Example 2:**
Input:
**Today's Date**: 2026-02-01
**Previous Chat History**:
user: I need to book a flight to London.
assistant: When are you planning to fly?

**Current Conversation Turn**:
user: I want to leave on the 15th and return on the 22nd. I strictly want to avoid overnight layovers.
assistant: Noted. I will filter for direct flights. Just a reminder: London is currently 8 hours ahead of your timezone.
Output:
{{
  "episodic_memories": [
    "2026-02-01: Planning flight to London | Details: The user needs to book a flight to London, wanting to leave on February 15th and return on February 22nd. They strictly want to avoid overnight layovers.",
    "2026-02-01: Assistant provides flight information | Details: The assistant noted the user's request, will filter for direct flights, and reminded them that London is currently 8 hours ahead of their timezone."
  ]
}}

Output format:
{{
  "episodic_memories": [
    "YYYY-MM-DD: Summary | Details: Detailed description capturing who, what, when, where, and why",
    "YYYY-MM-DD: Summary | Details: Detailed description capturing who, what, when, where, and why"
  ]
}}

Return a COMPLETE JSON object containing the episodic_memories in the format shown above. Do NOT return just the key name "episodic_memories" - you must return the entire JSON structure including both the key and its corresponding array value.
"""

# --- MEMREADER PROMPT FOR SEMANTIC MEMORY --- 
MEMREADER_PROMPT_SEMANTIC = """You are a dialogue memory generator. Your task is to write fragment semantic memories that capture only the NEW facts from the "Current Conversation" (do not repeat anything already covered in "Historical Memories").

Core principle:
Convert dialogue from first-person to third-person narration, preserving as much substantive information content from the original as possible, excluding only confirmed non-informative words.

What to preserve:
- All substantive information: people, events, times, places, causes, results, numbers, specific descriptions
- Original wording: Keep specific terms used in dialogue for titles, item names, activity descriptions, etc, numbers use Arabic numerals
- Emotional expressions: Retain explicit emotions and attitudes from original (like "happy", "worried", "likes"), but avoid adding subjective inferences not present in original

Style:
- Use English third-person narration.
- Write plain sentences (no lists/numbering/Markdown). Aim for 2-4 sentences, but allow longer to retain essential details.
- Use exact proper nouns as in the dialogue; do not replace/expand/infer names, organizations, or locations.
- Each memory should focus on one core fact or closely related fact group; avoid packing too many unrelated details into a single entry.

Semantic Memory Requirements:
- Focus on conceptual knowledge about people, places, objects, and concepts in the user's life
- Capture information about the user's personal relationships, places they frequent, objects they own or use, and concepts relevant to their life
- Include general preferences, characteristics, and properties that are specific to the user
- Each semantic memory should focus on one core concept or closely related concept group
- Emphasize knowledge that is unique to the user's personal experiences and life context

Input:
Historical memories (do not repeat): {previous_summary}
Conversation date: {conversation_date}
Current conversation: {new_dialogue}

Please generate semantic memories from the conversation. If the current conversation has no substantial new content, provide a minimal 1-2 sentence summary of the core topic or attitude expressed in this turn (do NOT output "no significant additions" or similar empty statements).

**IMPORTANT: JSON COMPLETENESS REQUIREMENT**
You MUST return a COMPLETE JSON object, not just the key name. The JSON must include both the key "semantic_memories" and its corresponding value (an array). Returning only "semantic_memories" without the complete JSON structure will cause an error.

Output format:
{{
  "semantic_memories": [
    "Semantic memory 1 content",
    "Semantic memory 2 content"
  ]
}}

Return the semantic memories in a json object format as shown above.
"""




# MEMORY_MANAGER_PROMPT = """You are a specialized Memory Manager Agent.
# Your role is to maintain the consistency and growth of a memory graph using the provided tools.

# [INPUTS]
# You will receive:
# 1. "New Facts": A list of atomic facts extracted from the latest user input.
# 2. "Existing Memories": A list of retrieved memory items, each with a simplified Integer ID (e.g., "0", "1", "2").
#    - These memories include those directly related to the new facts, as well as other related facts connected with these memories.
#    - They form a connected graph of information relevant to the new facts.

# [MANDATORY OUTPUT FORMAT]
# For every new fact you process, you MUST:
# 1. First generate a detailed thinking process
# 2. Then call the appropriate tool

# [THINKING PROCESS REQUIREMENTS]
# Your thinking process MUST include:
# - The specific new fact you're analyzing
# - Which existing memories are relevant (with their IDs)
# - How memories are connected through related facts
# - Your comparison and reasoning
# - Which operation you've decided to perform and why

# [OPERATIONS & GUIDELINES]
# Compare New Facts with Existing Memories and perform the following operations using the available tools. 
# DO NOT output raw JSON text. You MUST use the provided function tools.

# 1. **ADD (create_memory)**
#    - **Condition**: If a fact contains completely NEW information not present in Existing Memories.
#    - **Action**: Call `create_memory` with a concise summary of the facts, not just a simple concatenation.
#    - **Important**: Memory content should be a meaningful and concise summary.
  
# 2. **UPDATE (update_memory)**
#    - **Condition**: If a fact adds detail, corrects, or updates a specific Existing Memory.
#    - **Constraint**: You MUST use the Integer ID (e.g., "0") provided in the input as the `target_memory_id`.
#    - **Logic**: Merge the old content and new fact into a comprehensive statement, not just a simple concatenation.
#    - **Example**:
#      - Old (ID="0"): "User likes generic pizza."
#      - New Fact: "User loves pepperoni pizza."
#      - Action: `update_memory(target_memory_id="0", new_content="User loves pepperoni pizza", ...)`

# 3. **DELETE (delete_memory)**
#    - **Condition**: If a fact explicitly contradicts an Existing Memory (and the new fact is trusted), or if the memory is no longer valid.
#    - **Constraint**: Use the Integer ID (e.g., "1") as `target_memory_id`.

# 4. **INFER (infer_memory) [CRITICAL]**
#    - **Condition**: Look for higher-level insights. If combining "Memory A" and "Memory B" reveals a hidden connection or causality.
#    - **Action**: Call `infer_memory`.
#    - **Example**:
#      - Memory A (ID="2"): "User moved to Singapore."
#      - Memory B (ID="3"): "User bought a Type G power adapter."
#      - Inference: "User is preparing electronics for Singapore power standards."
#      - Action: `infer_memory(source_memory_ids=["2", "3"], inference_content="...")`

# 5. **NOOP (no_operation)**
#    - **Condition**: If the fact is redundant (already exactly covered by memory), similar to existing facts associated with the retrieved memories, or trivial.

# [STRICT ID RULES]
# - When calling `update_memory` or `delete_memory`, **ONLY** use the string integer IDs (e.g., "0", "1", "2") found in the [EXISTING MEMORIES] list.
# - **NEVER** invent a UUID or use an ID that is not in the provided list.
# """


# MEMORY_MANAGER_PROMPT = """You are a specialized Memory Manager Agent.
# Your role is to maintain the consistency and growth of a memory graph using the provided tools.

# [INPUTS]
# You will receive:
# 1. "New Facts": A list of atomic facts extracted from the latest user input.
# 2. "Existing Memories": A list of retrieved memory items, each with a simplified Integer ID (e.g., "0", "1", "2").
#    - These memories include those directly related to the new facts, as well as other related facts connected with these memories.
#    - They form a connected graph of information relevant to the new facts.

# [MANDATORY OUTPUT FORMAT]
# For every new fact you process, you MUST:
# 1. First generate a detailed thinking process
# 2. Then call the appropriate tool

# [THINKING PROCESS REQUIREMENTS]
# Your thinking process MUST include:
# - The specific new fact you're analyzing
# - Which existing memories are relevant (with their IDs)
# - How memories are connected through related facts
# - Your comparison and reasoning
# - Which operation you've decided to perform and why

# [OPERATIONS & GUIDELINES]
# Compare New Facts with Existing Memories and perform the following operations using the available tools. 
# DO NOT output raw JSON text. You MUST use the provided function tools.

# 1. **ADD (create_memory)**
#    - **Condition**: If a fact contains completely NEW information not present in Existing Memories.
#    - **Action**: Call `create_memory` with a concise summary of the facts, not just a simple concatenation.
#    - **Important**: Memory content should be a meaningful and concise summary.
  
# 2. **UPDATE (update_memory)**
#    - **Condition**: If a fact adds detail, corrects, or updates a specific Existing Memory.
#    - **Constraint**: You MUST use the Integer ID (e.g., "0") provided in the input as the `target_memory_id`.
#    - **Logic**: Merge the old content and new fact into a comprehensive statement, not just a simple concatenation.
#    - **Example**:
#      - Old (ID="0"): "User likes generic pizza."
#      - New Fact: "User loves pepperoni pizza."
#      - Action: `update_memory(target_memory_id="0", new_content="User loves pepperoni pizza", ...)`

# 3. **DELETE (delete_memory)**
#    - **Condition**: If a fact explicitly contradicts an Existing Memory (and the new fact is trusted), or if the memory is no longer valid.
#    - **Constraint**: Use the Integer ID (e.g., "1") as `target_memory_id`.

# 4. **INFER (infer_memory)**
#    - **Condition**: Look for **higher-level insights**, **patterns**, or **implicit goals**. Do not just store facts; synthesize them to find user habits or personality traits.
#    - **Action**: Call `infer_memory` to create a new insight that connects multiple memories.
#    - **Examples**:
#      - *Case 1: Causality/Goal*
#        - Input: Memory A (ID="2"): "User moved to Singapore." + Memory B (ID="3"): "User bought a Type G power adapter."
#        - Inference: "User is preparing electronics for Singapore power standards."
#        - Action: `infer_memory(source_memory_ids=["2", "3"], inference_content="User is preparing electronics for Singapore power standards")`
     
#      - *Case 2: Pattern Recognition*
#        - Input: Memory C (ID="5"): "User ordered Sichuan Hotpot." + Memory D (ID="6"): "User loves extra spicy curry."
#        - Inference: "User has a high tolerance for spicy food and prefers spicy cuisine."
#        - Action: `infer_memory(source_memory_ids=["5", "6"], inference_content="User has a high tolerance for spicy food and prefers spicy cuisine")`

# 5. **NOOP (no_operation)**
#    - **Condition**: If the fact is redundant (already exactly covered by memory), similar to existing facts associated with the retrieved memories, or trivial.

# [STRICT ID RULES]
# - When calling `update_memory` or `delete_memory`, **ONLY** use the string integer IDs (e.g., "0", "1", "2") found in the [EXISTING MEMORIES] list.
# - **NEVER** invent a UUID or use an ID that is not in the provided list.
# """

# --- NEW MEMREADER PROMPT WITH HISTORY SUPPORT ---
# MEMREADER_PROMPT_WITH_HISTORY = """You are a Personal Information Organizer, specialized in accurately storing facts, user memories, assistant memories and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts. This allows for easy retrieval and personalization in future interactions. Below are the types of information you need to focus on and the detailed instructions on how to handle the input conversation.

# Types of Information to Remember:

# 1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
# 2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
# 3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
# 4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
# 5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
# 6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
# 7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user and assistant share.

# Here are some few shot examples:

# Input:
# **Today's Date**: 2025-05-13
# **Previous Chat History**:
# user: I am planning a hiking trip to Yosemite National Park next weekend.
# assistant: That sounds wonderful! The views there are breathtaking. Who are you going with?

# **Current Conversation Turn**:
# user: I'm going with my brother, Mike. We hope to reach the Half Dome summit.
# Output: {{"facts" : [
#     {{"fact": "Going to Yosemite National Park with brother Mike", "details": ["Activity: Hiking", "Time: 2025-05-19 to 2025-05-25 (Next weekend)"]}},
#     {{"fact": "Hopes to summit Half Dome", "details": ["Location: Yosemite National Park"]}}
# ]}}

# Input:
# **Today's Date**: 2025-08-10
# **Previous Chat History**:
# assistant: I recall you mentioned being vegetarian. Is that still the case?

# **Current Conversation Turn**:
# user: Actually, no. I've started eating chicken recently for the protein, but I still avoid red meat.
# Output: {{"facts" : [
#     {{"fact": "Starts eating chicken", "details": ["Reason: For protein", "Status: Current diet update", "Avoid: Red meat", "Date: 2025-08-10"]}},
#     {{"fact": "Is no longer strictly vegetarian", "details": ["Reason: Eating chicken for the protein"]}}
# ]}}

# Input:
# **Today's Date**: 2025-10-01
# **Previous Chat History**:
# user: My laptop is broken, so I'm using the library computer.
# assistant: Oh no, I hope you can get it fixed. Are you working on your novel?

# **Current Conversation Turn**:
# user: Yes, I managed to write 2000 words for the 'The Lost City' draft despite the slow internet here.
# Output: {{"facts" : [
#     {{"fact": "Wrote 2000 words for 'The Lost City'", "details": ["Location: Library", "Device: Library computer"]}},
#     {{"fact": "Experiencing slow internet", "details": ["Location: Library", "Device: Library computer"]}}
# ]}}

# Input:
# **Today's Date**: 2025-11-15
# **Previous Chat History**:
# user: I'm looking for a gift for my wife.
# assistant: Does she have any specific hobbies?
# user: She loves outdoor sports.
# assistant: How about tennis gear?

# **Current Conversation Turn**:
# user: She already has plenty of that. She is really into rock climbing these days.
# Output: {{"facts" : [
#     {{"fact": "Wife is into rock climbing", "details": ["Interest: Outdoor sports"]}},
#     {{"fact": "Wife already has tennis gear", "details": []}}
# ]}}

# Input:
# **Today's Date**: 2026-02-01
# **Previous Chat History**:
# user: I need to book a flight to London.
# assistant: When are you planning to fly?

# **Current Conversation Turn**:
# user: I want to leave on the 15th and return on the 22nd. I strictly want to avoid overnight layovers.
# Output: {{"facts" : [
#     {{"fact": "Planning a flight to London", "details": ["Departure: 2026-02-15", "Return: 2026-02-22"]}},
#     {{"fact": "Strictly avoids overnight layovers", "details": ["Preference: Flight booking constraint"]}}
# ]}}

# Return the facts and preferences in a json format as shown above.

# Remember the following:
# - Today's date is {today_date}.
# - **Supplementary Details**: The `details` list must act as **METADATA** to supplement the fact (e.g., Time, Location, Price, Platform, Reason), **NOT** just splitting the fact's words. (e.g., If fact is "Bought apple", details should be ["Price: $1", "Store: Aldi"], NOT ["Action: Buy", "Object: Apple"]).
# - **Context Propagation**: Ensure every extracted fact is **self-contained**. If a shared context (e.g., location, platform, activity, or timeframe) is established anywhere in the input chunk or previous chat history, explicitly include it in the `details` of all relevant facts, even if not repeated in every sentence.
# - **Date Resolution**: ALWAYS resolve relative time expressions (e.g., "next weekend", "the 15th", "tomorrow") into absolute ISO dates (YYYY-MM-DD) based on Today's date provided in the input.
# - Do not return anything from the custom few shot example prompts provided above.
# - If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
# - Create the facts based on the user and assistant messages only. Do not pick anything from the system messages.
# - Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "facts", where each item has a "fact" string and a "details" list of strings.
# - When processing with previous chat history, leverage the context to extract more accurate and comprehensive facts, especially for anaphoric references and context-dependent information.

# Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user and facts about the assistant, if any, from the conversation and return them in the json format as shown above.
# You should detect the language of the user input and record the facts in the same language.
# """

MEMREADER_PROMPT_WITH_HISTORY = """You are a Personal Information Organizer, specialized in accurately storing facts, user memories, assistant memories and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts. This allows for easy retrieval and personalization in future interactions. Below are the types of information you need to focus on and the detailed instructions on how to handle the input conversation.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user and assistant share.

Here are some few shot examples:

Input:
**Today's Date**: 2025-05-13
**Previous Chat History**:
user: I am planning a hiking trip to Yosemite National Park next weekend.
assistant: That sounds wonderful! The views there are breathtaking. Who are you going with?

**Current Conversation Turn**:
user: I'm going with my brother, Mike. We hope to reach the Half Dome summit.
assistant: That sounds like a great challenge. I'll make a note that you are hiking Half Dome with Mike.
Output: {{"facts" : [
    {{"fact": "Going to Yosemite National Park with brother Mike", "details": ["Activity: Hiking", "Time: 2025-05-24 to 2025-05-25 (Next weekend)"]}},
    {{"fact": "Hopes to summit Half Dome", "details": ["Location: Yosemite National Park", "Intent: Hiking goal"]}}
]}}

Input:
**Today's Date**: 2025-08-10
**Previous Chat History**:
assistant: I recall you mentioned being vegetarian. Is that still the case?

**Current Conversation Turn**:
user: Actually, no. I've started eating chicken recently for the protein, but I still avoid red meat.
assistant: Understood. I've updated your dietary preferences to include chicken but exclude red meat.
Output: {{"facts" : [
    {{"fact": "Starts eating chicken", "details": ["Reason: For protein", "Status: Current diet update", "Avoid: Red meat", "Date: 2025-08-10"]}},
    {{"fact": "Is no longer strictly vegetarian", "details": ["Reason: Eating chicken for the protein"]}}
]}}

Input:
**Today's Date**: 2025-10-01
**Previous Chat History**:
user: My laptop is broken, so I'm using the library computer.
assistant: Oh no, I hope you can get it fixed. Are you working on your novel?

**Current Conversation Turn**:
user: Yes, I managed to write 2000 words for the 'The Lost City' draft despite the slow internet here.
assistant: That is impressive dedication! Writing 2000 words on a public computer is no small feat.
Output: {{"facts" : [
    {{"fact": "Wrote 2000 words for 'The Lost City'", "details": ["Location: Library", "Device: Library computer"]}},
    {{"fact": "Experiencing slow internet", "details": ["Location: Library", "Device: Library computer"]}}
]}}

Input:
**Today's Date**: 2025-11-15
**Previous Chat History**:
user: I'm looking for a gift for my wife.
assistant: Does she have any specific hobbies?
user: She loves outdoor sports.
assistant: How about tennis gear?

**Current Conversation Turn**:
user: She already has plenty of that. She is really into rock climbing these days.
assistant: Got it. Since she likes rock climbing, I recommend checking out the new indoor climbing gym downtown.
Output: {{"facts" : [
    {{"fact": "Wife is into rock climbing", "details": ["Interest: Outdoor sports"]}},
    {{"fact": "Wife already has tennis gear", "details": []}},
    {{"fact": "[Assistant] Recommends new indoor climbing gym downtown", "details": ["Reason: Matches wife's rock climbing interest"]}}
]}}

Input:
**Today's Date**: 2026-02-01
**Previous Chat History**:
user: I need to book a flight to London.
assistant: When are you planning to fly?

**Current Conversation Turn**:
user: I want to leave on the 15th and return on the 22nd. I strictly want to avoid overnight layovers.
assistant: Noted. I will filter for direct flights. Just a reminder: London is currently 8 hours ahead of your timezone.
Output: {{"facts" : [
    {{"fact": "Planning a flight to London", "details": ["Departure: 2026-02-15", "Return: 2026-02-22"]}},
    {{"fact": "Strictly avoids overnight layovers", "details": ["Preference: Flight booking constraint"]}},
    {{"fact": "[Assistant] London is 8 hours ahead of user's timezone", "details": ["Context: Timezone reminder"]}}
]}}

Return the facts and preferences in a json format as shown above.

Remember the following:
- Today's date is {today_date}.
- **Supplementary Details**: The `details` list must act as **METADATA** to supplement the fact (e.g., Time, Location, Price, Platform, Reason), **NOT** just splitting the fact's words.
- **Source Attribution**: If a fact or suggestion originates explicitly from the **assistant**, you MUST prefix the fact text with **"[Assistant]"**. (e.g., "[Assistant] Recommends checking out..."). Facts from the user do not need a prefix.
- **Context Propagation**: Ensure every extracted fact is **self-contained**. If a shared context (e.g., location, platform, activity, or timeframe) is established anywhere in the input chunk or previous chat history, explicitly include it in the `details` of all relevant facts, even if not repeated in every sentence.
- **Date Resolution**: ALWAYS resolve relative time expressions (e.g., "next weekend", "the 15th", "tomorrow") into absolute ISO dates (YYYY-MM-DD) based on Today's date provided in the input.
- Do not return anything from the custom few shot example prompts provided above.
- If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
- Create the facts based on the user and assistant messages only. Do not pick anything from the system messages.
- Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "facts", where each item has a "fact" string and a "details" list of strings.
- When processing with previous chat history, leverage the context to extract more accurate and comprehensive facts, especially for anaphoric references and context-dependent information.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user and facts about the assistant, if any, from the conversation and return them in the json format as shown above.
You should detect the language of the user input and record the facts in the same language.
"""

# --- TOOLS ---
MEMORY_TOOLS = [
    # Episodic Memory Tools
    {
        "type": "function",
        "function": {
            "name": "create_episodic_memory",
            "description": "Create a new event entry not currently in memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The episodic memory content, capturing specific events and interactions."}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trajectorize_episodic_memory",
            "description": "Combine multiple related events into a timeline with a timestamp range, drawing conclusions from patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_old_contents": {"type": "array", "items": {"type": "string"}, "description": "List of original contents of episodic memories that are related and will be combined."},
                    "new_content": {"type": "string", "description": "The new combined memory content that forms a timeline and draws conclusions."}
                },
                "required": ["source_old_contents", "new_content"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "delete_episodic_memory",
    #         "description": "Archive/Soft-delete an episodic memory if it is no longer valid.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "target_memory_id": {"type": "string", "description": "The simplified Integer ID (e.g., '1') of the episodic memory to delete, found in the [EXISTING EPISODIC MEMORIES] list."}
    #             },
    #             "required": ["target_memory_id"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "infer_episodic_memory",
            "description": "Look for higher-level insights from multiple episodic memories. Create a new insight that connects multiple episodic memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_old_contents": {"type": "array", "items": {"type": "string"}, "description": "List of original contents of episodic memories acting as premises, found in the [EXISTING EPISODIC MEMORIES] list."},
                    "inference_content": {"type": "string", "description": "The inferred episodic memory content, connecting multiple episodic memories."}
                },
                "required": ["source_old_contents", "inference_content"]
            }
        }
    },
    # Semantic Memory Tools
    {
        "type": "function",
        "function": {
            "name": "create_semantic_memory",
            "description": "Create a NEW semantic memory capturing conceptual knowledge about people, places, objects, and concepts in the user's life.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The semantic memory content, capturing conceptual knowledge about the user's life."}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_semantic_memory",
            "description": "Update an existing semantic memory about conceptual knowledge and concepts in the user's life with new content based on the old memory content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_content": {"type": "string", "description": "The original content of the semantic memory about the user's life to update."},
                    "new_content": {"type": "string", "description": "The updated semantic memory content about the user's life."}
                },
                "required": ["old_content", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_semantic_memory",
            "description": "Archive/Soft-delete a semantic memory if it is no longer valid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_content": {"type": "string", "description": "The original content of the semantic memory to delete."}
                },
                "required": ["old_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "infer_semantic_memory",
            "description": "Look for higher-level insights from multiple semantic memories about the user's life. Create a new insight that connects multiple semantic memories about the user's life.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_old_contents": {"type": "array", "items": {"type": "string"}, "description": "List of original contents of semantic memories about the user's life acting as premises, found in the [EXISTING SEMANTIC MEMORIES] list."},
                    "inference_content": {"type": "string", "description": "The inferred semantic memory content about the user's life, connecting multiple semantic memories."}
                },
                "required": ["source_old_contents", "inference_content"]
            }
        }
    },
    # Common Tool
    {
        "type": "function",
        "function": {
            "name": "no_operation",
            "description": "No action needed if the fact is redundant (already exactly covered by memory or its associated facts).",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "The reason for no operation."}},
                "required": ["reason"]
            }
        }
    }
]

CORE_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "core_memory_rewrite",
            "description": "Re-generate the entire Core Memory profile or a major section to ensure flow and consistency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The complete rewritten structured profile content."}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": "Update a specific section or paragraph in the Core Memory when details change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_text": {"type": "string", "description": "The exact text segment to be replaced."},
                    "new_text": {"type": "string", "description": "The new text segment to replace the old one."}
                },
                "required": ["old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": "Add a completely new section or distinct information to the end of the Core Memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The additional text content to append."}
                },
                "required": ["content"]
            }
        }
    }
]

# --- UTILS ---
def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    return llm_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

@dataclass
class MilvusConfig:
    """Milvus配置类（兼容旧代码）"""
    uri: str = os.getenv("MILVUS_URI")
    user_name: str = os.getenv("MILVUS_USER_NAME")
    # password: str = os.getenv("MILVUS_PASSWORD")
    db_name: str = os.getenv("MILVUS_DB_NAME", "default")
    dimension: int = 1536
    
    def to_vector_db_config(self, vector_db_type: str = "milvus") -> VectorDBConfig:
        """转换为VectorDBConfig"""
        # 确保vector_db_type是字符串类型
        if not isinstance(vector_db_type, str):
            vector_db_type = "milvus"  # 默认使用milvus
        
        # 根据vector_db_type选择不同的URL
        if vector_db_type == "qdrant":
            uri = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            user_name = ""
            password = ""
        else:
            uri = self.uri
            api_key = ""
            user_name = self.user_name
            password = os.getenv("MILVUS_PASSWORD")
        
        return VectorDBConfig(
            uri=uri,
            user_name=user_name,
            password=password,
            api_key=api_key,
            db_name=self.db_name,
            dimension=self.dimension,
            vector_db_type=vector_db_type
        )

# ==========================================
# 1. Pipeline Class
# ==========================================
class MemoryPipeline:
    def __init__(self, config=None, vector_db_type="milvus", clear_db=False, mode='eval', dataset_name=""):
        """初始化MemoryPipeline
        
        Args:
            config: MilvusConfig或VectorDBConfig实例，如果为None则使用默认配置
            vector_db_type: 指定使用的向量数据库类型，支持"milvus"或"qdrant"
            clear_db: 是否清空数据库，默认为False
            dataset_name: 数据集名称，用于集合名称后缀，默认为空
        """
        # 如果没有提供配置，创建默认配置
        if config is None:
            config = MilvusConfig()
        
        self.config = config
        
        # 转换为VectorDBConfig
        if hasattr(config, 'to_vector_db_config'):
            vector_db_config = config.to_vector_db_config(vector_db_type=vector_db_type)
        else:
            # 如果已经是VectorDBConfig实例，直接使用
            vector_db_config = config
        
        # 使用工厂类创建向量数据库客户端
        self.client = VectorDBFactory.create_db(vector_db_config)
        
        # 根据模式和数据集名称设置集合名称
        base_suffix = "_test" if mode == 'test' else ""
        dataset_suffix = f"_{dataset_name}" if dataset_name else ""
        full_suffix = f"{base_suffix}{dataset_suffix}"
        
        self.semantic_col = f"semantic_memories{full_suffix}_v1"  # semantic memory
        self.episodic_col = f"episodic_memories{full_suffix}_v1"  # episodic memory
        
        self.dim = vector_db_config.dimension  # Save dimension as instance variable
        # 初始化操作次数计数器
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0, "TRAJECTORIZE": 0}
        # 添加带历史支持的memreader prompt
        self.MEMREADER_PROMPT_WITH_HISTORY = MEMREADER_PROMPT_WITH_HISTORY
        # 初始化core memory为空
        self.core_memory = ""
        self._init_collections(clear_db=clear_db)

    def _init_collections(self, clear_db=False):
        dim = self.config.dimension
        
        # 如果需要清空数据库，先删除所有集合
        if clear_db:
            print("正在清空数据库...")
            # 直接删除集合，不检查存在性
            self.client.drop_collection(self.semantic_col)
            self.client.drop_collection(self.episodic_col)
            print("数据库清空完成.")
        
        # 检查并创建集合
        
        # 处理 semantic memories 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.semantic_col):
                # 创建完整的schema
                s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                s.add_field("memory_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
                s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
                s.add_field("content", self.client.DataType.VARCHAR, max_length=65535)
                s.add_field("user_id", self.client.DataType.VARCHAR, max_length=64)
                s.add_field("status", self.client.DataType.VARCHAR, max_length=16)
                s.add_field("created_at", self.client.DataType.INT64)
                s.add_field("updated_at", self.client.DataType.INT64)
                s.add_field("relations", self.client.DataType.JSON) 
                
                # 创建集合
                self.client.create_collection(self.semantic_col, schema=s)
                print(f"Collection '{self.semantic_col}' created.")
                
                # 直接创建索引，不检查索引是否存在
                # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
                try:
                    print(f"为集合 '{self.semantic_col}' 创建索引...")
                    # 创建向量索引
                    idx_params = self.client.prepare_index_params()
                    idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                    self.client.create_index(self.semantic_col, index_params=idx_params)
                    # 创建文本字段索引，支持关键词检索
                    try:
                        # 为content字段创建标量索引
                        text_idx_params = self.client.prepare_index_params()
                        text_idx_params.add_index(field_name="content", index_type="INVERTED", params={})
                        self.client.create_index(self.semantic_col, index_params=text_idx_params)
                    except Exception as text_idx_error:
                        # 忽略文本索引错误，因为某些Milvus版本可能不支持
                        print(f"创建文本索引失败 (忽略): {text_idx_error}")
                    print(f"集合 '{self.semantic_col}' 的索引创建成功或已存在")
                except Exception as e:
                    print(f"创建索引失败: {e}")
            else:
                print(f"Collection '{self.semantic_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.semantic_col)
            print(f"Collection '{self.semantic_col}' created or exists.")
        
        # 处理 episodic memories 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.episodic_col):
                # 创建完整的schema
                s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                s.add_field("memory_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
                s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
                s.add_field("content", self.client.DataType.VARCHAR, max_length=65535)
                s.add_field("user_id", self.client.DataType.VARCHAR, max_length=64)
                s.add_field("status", self.client.DataType.VARCHAR, max_length=16)
                s.add_field("created_at", self.client.DataType.INT64)
                s.add_field("updated_at", self.client.DataType.INT64)
                s.add_field("relations", self.client.DataType.JSON)
                
                # 创建集合
                self.client.create_collection(self.episodic_col, schema=s)
                print(f"Collection '{self.episodic_col}' created.")
                
                # 直接创建索引，不检查索引是否存在
                # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
                try:
                    print(f"为集合 '{self.episodic_col}' 创建索引...")
                    # 创建向量索引
                    idx_params = self.client.prepare_index_params()
                    idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                    self.client.create_index(self.episodic_col, index_params=idx_params)
                    # 创建文本字段索引，支持关键词检索
                    try:
                        # 为content字段创建标量索引
                        text_idx_params = self.client.prepare_index_params()
                        text_idx_params.add_index(field_name="content", index_type="INVERTED", params={})
                        self.client.create_index(self.episodic_col, index_params=text_idx_params)
                    except Exception as text_idx_error:
                        # 忽略文本索引错误，因为某些Milvus版本可能不支持
                        print(f"创建文本索引失败 (忽略): {text_idx_error}")
                    print(f"集合 '{self.episodic_col}' 的索引创建成功或已存在")
                except Exception as e:
                    print(f"创建索引失败: {e}")
            else:
                print(f"Collection '{self.episodic_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.episodic_col)
            print(f"Collection '{self.episodic_col}' created or exists.")
        

        
        # 直接加载所有集合，不进行复杂的错误处理
        print("Loading collections into memory...")
        
        # 加载集合（Qdrant 不需要显式加载）
        if hasattr(self.client, 'load_collection'):
            # 为每个集合创建索引后直接加载
            print(f"加载集合 '{self.semantic_col}'...")
            self.client.load_collection(self.semantic_col)
            
            print(f"加载集合 '{self.episodic_col}'...")
            self.client.load_collection(self.episodic_col)
            
            print("All collections loaded successfully.")

    # --- Step 1: Extract ---
    def step_extract(self, session_or_text, extract_mode: str = "whole", timestamp: int = None, max_history_turns: int = 5) -> Dict:
        """
        从对话中提取事实
        
        Args:
            session_or_text: 对话会话，可以是原始session list或对话文本
            extract_mode: 提取模式，可选值：
                - "whole": 对整个chunk进行提取
                - "turn": 按轮次提取，每轮user-assistant对话单独提取，并附上chat history
            timestamp: 时间戳，可选，默认使用当前时间
            max_history_turns: 聊天历史的最大轮数，仅在extract_mode="turn"时生效
        
        Returns:
            包含提取事实的字典
        """
        # print(f"\n👀 [1. Extract] Processing...")
        
        # 如果没有提供timestamp，使用当前时间
        if timestamp is None:
            timestamp = int(time.time())
        
        # 处理session_or_text为list的情况
        # 注意：当extract_mode="turn"时，我们需要使用原始的session_or_text列表进行按轮次处理
        if isinstance(session_or_text, list):
            if extract_mode == "turn":
                # 当按轮次处理时，直接使用原始的session_or_text列表
                # 这样在_extract_memories函数中可以正确按轮次处理
                chunk_text = ""  # 不需要转换为字符串
            else:
                # 当对整个chunk处理时，将session list转换为字符串
                chunk_text = parse_messages(session_or_text)
        else:
            # 如果输入已经是字符串，直接使用
            chunk_text = session_or_text
        
        # 提取episodic和semantic记忆
        result = self._extract_memories(chunk_text, timestamp, extract_mode, max_history_turns, session_or_text)
        return result
    
    def _extract_memories(self, chunk_text: str, timestamp: int, extract_mode: str, max_history_turns: int, session_or_text) -> Dict:
        """
        提取不同类型的记忆
        
        Args:
            chunk_text: 完整的对话文本
            timestamp: 时间戳
            extract_mode: 提取模式
            max_history_turns: 最大历史轮数
            session_or_text: 原始会话或文本
                - 当 extract_mode="turn" 时，需要提供原始的 session list 以按轮次处理
                - 当 extract_mode="whole" 时，可以是会话列表或文本
        
        Returns:
            包含不同类型记忆的字典
        """
        # 确保 session_or_text 是列表格式，以便按轮次处理
        if extract_mode == "turn" and not isinstance(session_or_text, list):
            print(f"⚠️  警告: extract_mode='turn' 时需要提供原始的 session list，当前输入为文本格式，将回退到 'whole' 模式")
            extract_mode = "whole"
        episodic_memories = []
        semantic_memories = []
        all_facts = []
        chat_history = []
        
        # 按轮次提取episodic memory
        if extract_mode == "turn" and isinstance(session_or_text, list):
            try:
                # 遍历session list，智能构建turn
                i = 0
                turn_count = 0
                while i < len(session_or_text):
                    # 跳过system消息
                    while i < len(session_or_text) and session_or_text[i].get("role") == "system":
                        i += 1
                    
                    # 确保还有消息
                    if i >= len(session_or_text):
                        break
                    
                    # 找到user消息
                    if session_or_text[i].get("role") == "user":
                        # 构建当前turn，包含user消息
                        turn = [session_or_text[i]]
                        i += 1
                        
                        # 如果下一条消息是assistant消息，添加到当前turn
                        if i < len(session_or_text) and session_or_text[i].get("role") == "assistant":
                            turn.append(session_or_text[i])
                            i += 1
                        
                        # 验证turn是否有效
                        if not turn:
                            continue
                        
                        # 将turn转换为文本格式
                        turn_text = parse_messages(turn)
                        
                        # 添加当前turn到chat history
                        chat_history.append(turn)
                        turn_count += 1
                        
                        # 构建聊天历史，使用当前turn之前的max_history_turns轮对话
                        history_turns = chat_history[:-1][-max_history_turns:]  # 最近max_history_turns轮历史
                        history_text = parse_messages([msg for turn in history_turns for msg in turn])
                        
                        # 对单轮对话提取episodic事实，传递timestamp和chat_history参数
                        turn_facts = self._extract_single_turn(turn_text, timestamp, history_text, memory_type="episodic")
                        
                        # 为每个事实添加轮次信息和chat history引用
                        for fact in turn_facts:
                            fact["has_history"] = len(history_text) > 0
                            fact["history_turns"] = len(history_turns)  # 聊天历史的轮数
                            fact["memory_type"] = "episodic"  # 标记为episodic memory
                        
                        all_facts.extend(turn_facts)
                        
                        # 为每轮对话创建episodic memory
                        episodic_memory = {
                            "content": turn_text,
                            "chat_history": history_text,
                            "timestamp": timestamp,
                            "facts": turn_facts
                        }
                        episodic_memories.append(episodic_memory)
                    else:
                        # 如果当前消息不是user消息，跳过
                        i += 1
                
                # 打印处理统计信息
                print(f"   ✅ 按轮次处理完成，共处理 {turn_count} 轮对话")
            except Exception as e:
                print(f"   ⚠️ 按轮次处理session失败，回退到whole模式: {e}")
        
        # 提取semantic memory（从整个session中）
        # 如果chunk_text为空（当extract_mode="turn"时），我们需要从session_or_text中构建完整的对话文本
        if not chunk_text and isinstance(session_or_text, list):
            semantic_chunk_text = parse_messages(session_or_text)
        else:
            semantic_chunk_text = chunk_text
        
        # 对整个session提取semantic事实，传递timestamp参数
        semantic_facts = self._extract_single_turn(semantic_chunk_text, timestamp, memory_type="semantic")
        for fact in semantic_facts:
            fact["memory_type"] = "semantic"  # 标记为semantic memory
        all_facts.extend(semantic_facts)
        
        # 创建semantic memory
        semantic_memory = {
            "content": semantic_chunk_text,
            "timestamp": timestamp,
            "facts": semantic_facts
        }
        semantic_memories.append(semantic_memory)
        
        return {
            "chunk_id": str(uuid.uuid4()),
            "chunk_text": semantic_chunk_text,  # 使用构建好的semantic_chunk_text，确保它在extract_mode="turn"时也能正确显示完整的对话文本
            "new_facts": all_facts,
            "episodic_memories": episodic_memories,
            "semantic_memories": semantic_memories,
            "timestamp": timestamp,
            "chat_history": chat_history
        }
    
    def _extract_single_turn(self, text: str, timestamp: int = None, chat_history: str = "", memory_type: str = "episodic") -> List[Dict]:
        """
        对单个文本片段提取事实
        
        Args:
            text: 要提取事实的文本
            timestamp: 时间戳，可选，默认使用当前时间
            chat_history: 之前的对话历史，用于提供上下文
            memory_type: 记忆类型，可选值：
                - "episodic": 提取情景记忆
                - "semantic": 提取语义记忆
            
        Returns:
            提取到的事实列表
        """
        try:
            # 将timestamp转换为日期字符串（YYYY-MM-DD）
            conversation_date = ""
            if timestamp:
                from datetime import datetime
                conversation_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            
            # 根据记忆类型选择相应的prompt
            if memory_type == "episodic":
                prompt = MEMREADER_PROMPT_EPISODIC
                formatted_prompt = prompt.format(previous_summary=chat_history, conversation_date=conversation_date, new_dialogue=text)
                expected_key = "episodic_memories"
            else:  # semantic
                prompt = MEMREADER_PROMPT_SEMANTIC
                formatted_prompt = prompt.format(previous_summary=chat_history, conversation_date=conversation_date, new_dialogue=text)
                expected_key = "semantic_memories"
            
            response = llm_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                        {"role": "system", "content": formatted_prompt}
                        ],
                response_format={"type": "json_object"}, temperature=0
            )
            
            # 获取响应内容
            response_content = response.choices[0].message.content
            print(f"Debug: Received response content: {response_content[:200]}...")
            
            # 清理和解析JSON
            cleaned_content = response_content.strip()
            # 移除可能的Markdown代码块标记
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:].strip()
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3].strip()
            
            # 检查是否为不完整的JSON（只包含键名）
            if cleaned_content == f'"{expected_key}"':
                # 只返回了键名，没有值，返回原始文本
                print(f"Extraction failed: Only key name returned without value - '{cleaned_content}'")
                return [{"text": text, "timestamp": timestamp}]
            
            # 解析JSON
            memory_data = json.loads(cleaned_content)
            print("Debug: JSON parsing succeeded")
            
            # 检查是否包含预期的键
            if expected_key in memory_data:
                memories = memory_data.get(expected_key, [])
                
                # 将memories转换为事实格式
                facts = []
                for memory in memories:
                    fact = {
                        "text": memory,
                        "timestamp": timestamp
                    }
                    facts.append(fact)
                
                return facts
            else:
                # 缺少预期的键，返回原始文本
                print(f"Extraction failed: Missing expected key '{expected_key}'")
                return [{"text": text, "timestamp": timestamp}]
        except json.JSONDecodeError as e:
            # JSON解析失败，返回原始文本
            print(f"Extraction failed: JSON parsing error - {e}")
            return [{"text": text, "timestamp": timestamp}]
        except Exception as e:
            # 其他错误，返回原始文本
            print(f"Extraction failed: {e}")
            return [{"text": text, "timestamp": timestamp}]


    # --- Step 2: Retrieve ---    
    def step_retrieve(self, extract_result: Dict, limit: int = 3, user_id: str = 'default', similarity_threshold: float = None) -> List[Dict]:
        new_facts = extract_result['new_facts']
        if not new_facts: return []
        
        print(f"🔍 [2. Retrieve] Searching Memories for {len(new_facts)} facts...")
        context_bundles = []

        for fact in new_facts:
            query_vec = get_embedding(fact['text'])
            # 根据memory_type决定检索哪个集合
            memory_type = fact.get('memory_type', 'semantic')
            collection_name = self.episodic_col if memory_type == 'episodic' else self.semantic_col
            
            # 添加user_id过滤，确保只检索当前用户的记忆
            res = self.client.search(
                collection_name, [query_vec], filter=f"status == 'active' and user_id == '{user_id}'", limit=limit,
                output_fields=["content", "memory_id", "created_at"],
                similarity_threshold=similarity_threshold
            )
            candidates = []
            if res and res[0]:
                for hit in res[0]:
                    candidates.append(hit['entity'])
            
            # 将memory_type添加到每个memory对象中
            for mem in candidates:
                mem['memory_type'] = memory_type  # 添加记忆类型
            
            context_bundles.append({
                "new_fact": fact,
                "candidates": candidates,
                "memory_type": memory_type
            })
            
        return context_bundles

    # --- Step 3: Manage Episodic Memory ---
    def step_manage_episodic_memory(self, episodic_memories: List[Dict], retrieved_memories: List[Dict], user_id: str = 'default') -> List[Dict]:
        """
        管理episodic memory
        
        Args:
            episodic_memories: 新提取的episodic memory列表
            retrieved_memories: 检索到的相关旧episodic memory
            user_id: 用户标识
        
        Returns:
            记忆操作决策列表
        """
        print(f"🧠 [3.1 Manage Episodic Memory] Processing {len(episodic_memories)} memories...")
        decisions = []
        
        for episodic_memory in episodic_memories:
            # 提取episodic memory中的事实和时间戳
            facts = episodic_memory.get('facts', [])
            if not facts:
                continue
            
            # 获取对话的时间戳，用于记忆操作
            conversation_timestamp = episodic_memory.get('timestamp', int(time.time()))
            
            # 构造候选记忆字符串
            candidates_str = ""
            
            if retrieved_memories:
                for mem in retrieved_memories:
                    candidates_str += f"- Content: {mem['content']}\n"
            else:
                candidates_str = "(No relevant episodic memories found. Treat as new topic.)"
            
            # 构造prompt和用户输入
            system_msg = EPISODIC_MEMORY_PROMPT
            fact_texts = [fact['text'] for fact in facts]
            user_content = f"""
            [New Episodic Facts]
            {json.dumps(fact_texts, ensure_ascii=False)}
            
            [EXISTING EPISODIC MEMORIES]
            {candidates_str}
            """
            
            # 调用LLM进行决策
            try:
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content}
                    ],
                    tools=MEMORY_TOOLS,
                    tool_choice="required",
                    temperature=0
                )
                
                if response.choices and response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        
                        decision = {"action": "NOOP"}
                        
                        if func_name == "create_episodic_memory":
                            decision.update({
                                "action": "ADD", 
                                "summary": args.get("content", ""), 
                                "facts_to_link": [],
                                "user_id": user_id,
                                "memory_type": "episodic",
                                "created_at": conversation_timestamp
                            })
                        elif func_name == "trajectorize_episodic_memory":
                            if "source_old_contents" in args:
                                source_old_contents = args["source_old_contents"]
                                if not isinstance(source_old_contents, list):
                                    source_old_contents = [source_old_contents]
                                # 遍历 retrieved_memories，找到 content 与 source_old_contents 匹配的记忆
                                real_source_ids = []
                                for old_content in source_old_contents:
                                    for mem in retrieved_memories:
                                        if mem.get("content") == old_content:
                                            real_source_ids.append(mem.get("memory_id"))
                                            break
                                if real_source_ids:
                                    decision.update({
                                        "action": "TRAJECTORIZE", 
                                        "source_ids": real_source_ids, 
                                        "summary": args.get("new_content", ""), 
                                        "facts_to_link": [],
                                        "user_id": user_id,
                                        "memory_type": "episodic"
                                    })
                        elif func_name == "delete_episodic_memory":
                            if "old_content" in args:
                                old_content = args["old_content"]
                                # 遍历 retrieved_memories，找到 content 与 old_content 匹配的记忆
                                real_tid = None
                                orig_created = conversation_timestamp
                                for mem in retrieved_memories:
                                    if mem.get("content") == old_content:
                                        real_tid = mem.get("memory_id")
                                        orig_created = mem.get("created_at", conversation_timestamp)
                                        break
                                if real_tid:
                                    decision.update({
                                        "action": "DELETE", 
                                        "target_id": real_tid, 
                                        "facts_to_link": [], 
                                        "orig_created": orig_created,
                                        "user_id": user_id,
                                        "memory_type": "episodic"
                                    })
                        elif func_name == "infer_episodic_memory":
                            if "source_old_contents" in args:
                                source_old_contents = args["source_old_contents"]
                                if not isinstance(source_old_contents, list):
                                    source_old_contents = [source_old_contents]
                                # 遍历 retrieved_memories，找到 content 与 source_old_contents 匹配的记忆
                                real_source_ids = []
                                for old_content in source_old_contents:
                                    for mem in retrieved_memories:
                                        if mem.get("content") == old_content:
                                            real_source_ids.append(mem.get("memory_id"))
                                            break
                                if real_source_ids:
                                    decision.update({
                                        "action": "INFER", 
                                        "source_ids": real_source_ids, 
                                        "summary": args.get("inference_content", ""), 
                                        "facts_to_link": [],
                                        "user_id": user_id,
                                        "memory_type": "episodic"
                                    })
                        
                        if decision["action"] != "NOOP" or "reason" in decision:
                            decisions.append(decision)
            except Exception as e:
                print(f"   ⚠️ Episodic memory management error: {e}")
        
        return decisions
    
    # --- Step 3: Manage Core Memory ---
    def step_manage_core_memory(self, episodic_memories: List[Dict], semantic_memories: List[Dict]) -> str:
        """
        管理Core Memory
        
        Args:
            episodic_memories: 新的episodic memory列表
            semantic_memories: 新的semantic memory列表
        
        Returns:
            更新后的Core Memory文本
        """
        print(f"🧠 [3.3 Manage Core Memory] Processing...")
        
        # 提取新记忆中的关键信息
        new_episodic_information = []
        for mem in episodic_memories:
            content = mem.get('content', '')
            if content:
                new_episodic_information.append(content)
        
        new_semantic_information = []
        for mem in semantic_memories:
            content = mem.get('content', '')
            if content:
                new_semantic_information.append(content)
        
        # 构造用户输入
        episodic_info_str = "\n".join(new_episodic_information)
        semantic_info_str = "\n".join(new_semantic_information)
        
        # 限制输入长度，确保不超过模型的上下文窗口
        MAX_INPUT_LENGTH = 10000  # 设置最大输入长度
        
        # 构建基础输入
        base_content = f"""
        [Current Core Memory]
        {self.core_memory}
        
        [New Episodic Memories]
        """
        
        # 计算剩余可用长度
        remaining_length = MAX_INPUT_LENGTH - len(base_content)
        
        # 限制episodic和semantic记忆的长度
        if len(episodic_info_str) > remaining_length * 0.7:
            # 优先保留最新的记忆
            episodic_lines = episodic_info_str.split('\n')
            # 计算每行会占用的平均长度
            avg_line_length = len(episodic_info_str) / len(episodic_lines)
            # 计算可以保留的行数
            max_lines = int((remaining_length * 0.7) / avg_line_length)
            # 保留最新的记忆（假设后面的记忆更新）
            episodic_info_str = "\n".join(episodic_lines[-max_lines:])
        
        # 计算剩余长度
        remaining_length_after_episodic = remaining_length - len(episodic_info_str)
        
        if len(semantic_info_str) > remaining_length_after_episodic:
            # 优先保留最新的记忆
            semantic_lines = semantic_info_str.split('\n')
            # 计算每行会占用的平均长度
            avg_line_length = len(semantic_info_str) / len(semantic_lines)
            # 计算可以保留的行数
            max_lines = int(remaining_length_after_episodic / avg_line_length)
            # 保留最新的记忆（假设后面的记忆更新）
            semantic_info_str = "\n".join(semantic_lines[-max_lines:])
        
        # 构建最终输入
        user_content = f"""
        [Current Core Memory]
        {self.core_memory}
        
        [New Episodic Memories]
        {episodic_info_str}
        
        [New Semantic Memories]
        {semantic_info_str}
        """
        
        # 调用LLM进行决策
        try:
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CORE_MEMORY_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                tools=CORE_MEMORY_TOOLS,
                tool_choice="required",
                temperature=0.1  # 适当调整温度，允许模型有一定的创造性
            )
            
            if response.choices and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    if func_name == "core_memory_rewrite":
                        new_core_memory = args.get("content", "").strip()
                        if new_core_memory:
                            self.core_memory = new_core_memory
                            print(f"   ✅ Core Memory REWRITE: Reorganized entire block")
                        else:
                            print(f"   ⚠️ Core Memory REWRITE: Empty content")
                            
                    elif func_name == "core_memory_replace":
                        old_fragment = args.get("old_text", "").strip()
                        new_fragment = args.get("new_text", "").strip()
                        if old_fragment and new_fragment:
                            if old_fragment in self.core_memory:
                                self.core_memory = self.core_memory.replace(old_fragment, new_fragment)
                                print(f"   ✅ Core Memory REPLACE: Updated specific fragment")
                            else:
                                print(f"   ⚠️ Core Memory REPLACE: Old fragment not found")
                        else:
                            print(f"   ⚠️ Core Memory REPLACE: Empty old or new fragment")
                            
                    elif func_name == "core_memory_append":
                        new_content = args.get("content", "").strip()
                        if new_content:
                            if self.core_memory:
                                self.core_memory += f"\n\n{new_content}"
                            else:
                                self.core_memory = new_content
                            print(f"   ✅ Core Memory APPEND: Added new information")
                        else:
                            print(f"   ⚠️ Core Memory APPEND: Empty content")
                            
            else:
                # 处理空响应
                print(f"   ⚠️ Core Memory: No tool calls in response")
                # 尝试使用默认操作
                if new_episodic_information or new_semantic_information:
                    # 简单地将新信息添加到Core Memory
                    new_info = "\n".join(new_episodic_information + new_semantic_information)
                    if self.core_memory:
                        self.core_memory += f"\n\n{new_info[:500]}"  # 限制添加的长度
                    else:
                        self.core_memory = new_info[:500]  # 限制添加的长度
                    print(f"   ✅ Core Memory: Added new information using default operation")
        
        except Exception as e:
            print(f"   ⚠️ Core Memory management error: {e}")
        
        print(f"   📝 Updated Core Memory: {self.core_memory[:100]}...")
        return self.core_memory
    
    # --- Step 3: Manage Semantic Memory ---
    def step_manage_semantic_memory(self, semantic_memories: List[Dict], retrieved_memories: List[Dict], user_id: str = 'default') -> List[Dict]:
        """
        管理semantic memory
        
        Args:
            semantic_memories: 新提取的semantic memory列表
            retrieved_memories: 检索到的相关旧semantic memory
            user_id: 用户标识
        
        Returns:
            记忆操作决策列表
        """
        print(f"🧠 [3.2 Manage Semantic Memory] Processing {len(semantic_memories)} memories...")
        decisions = []
        
        for semantic_memory in semantic_memories:
            # 提取semantic memory中的事实和时间戳
            facts = semantic_memory.get('facts', [])
            if not facts:
                continue
            
            # 获取对话的时间戳，用于记忆操作
            conversation_timestamp = semantic_memory.get('timestamp', int(time.time()))
            
            # 构造候选记忆字符串
            candidates_str = ""
            
            if retrieved_memories:
                for mem in retrieved_memories:
                    candidates_str += f"- Content: {mem['content']}\n"
            else:
                candidates_str = "(No relevant semantic memories found. Treat as new topic.)"
            
            # 构造prompt和用户输入
            system_msg = SEMANTIC_MEMORY_PROMPT
            fact_texts = [fact['text'] for fact in facts]
            user_content = f"""
            [New Semantic Facts]
            {json.dumps(fact_texts, ensure_ascii=False)}
            
            [EXISTING SEMANTIC MEMORIES]
            {candidates_str}
            """
            
            # 调用LLM进行决策
            try:
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content}
                    ],
                    tools=MEMORY_TOOLS,
                    tool_choice="required",
                    temperature=0
                )
                
                if response.choices and response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        
                        decision = {"action": "NOOP"}
                        
                        if func_name == "create_semantic_memory":
                            decision.update({
                                "action": "ADD", 
                                "summary": args.get("content", ""), 
                                "facts_to_link": [],
                                "user_id": user_id,
                                "memory_type": "semantic",
                                "created_at": conversation_timestamp
                            })
                        elif func_name == "update_semantic_memory":
                            if "old_content" in args:
                                old_content = args["old_content"]
                                # 遍历 retrieved_memories，找到 content 与 old_content 匹配的记忆
                                real_tid = None
                                orig_created = conversation_timestamp
                                for mem in retrieved_memories:
                                    if mem.get("content") == old_content:
                                        real_tid = mem.get("memory_id")
                                        orig_created = mem.get("created_at", conversation_timestamp)
                                        break
                                if real_tid:
                                    decision.update({
                                        "action": "UPDATE", 
                                        "target_id": real_tid, 
                                        "new_content": args.get("new_content", ""), 
                                        "facts_to_link": [], 
                                        "orig_created": orig_created,
                                        "user_id": user_id,
                                        "memory_type": "semantic"
                                    })
                        elif func_name == "delete_semantic_memory":
                            if "old_content" in args:
                                old_content = args["old_content"]
                                # 遍历 retrieved_memories，找到 content 与 old_content 匹配的记忆
                                real_tid = None
                                orig_created = conversation_timestamp
                                for mem in retrieved_memories:
                                    if mem.get("content") == old_content:
                                        real_tid = mem.get("memory_id")
                                        orig_created = mem.get("created_at", conversation_timestamp)
                                        break
                                if real_tid:
                                    decision.update({
                                        "action": "DELETE", 
                                        "target_id": real_tid, 
                                        "facts_to_link": [], 
                                        "orig_created": orig_created,
                                        "user_id": user_id,
                                        "memory_type": "semantic"
                                    })
                        elif func_name == "infer_semantic_memory":
                            if "source_old_contents" in args:
                                source_old_contents = args["source_old_contents"]
                                if not isinstance(source_old_contents, list):
                                    source_old_contents = [source_old_contents]
                                # 遍历 retrieved_memories，找到 content 与 source_old_contents 匹配的记忆
                                real_source_ids = []
                                for old_content in source_old_contents:
                                    for mem in retrieved_memories:
                                        if mem.get("content") == old_content:
                                            real_source_ids.append(mem.get("memory_id"))
                                            break
                                if real_source_ids:
                                    decision.update({
                                        "action": "INFER", 
                                        "source_ids": real_source_ids, 
                                        "summary": args.get("inference_content", ""), 
                                        "facts_to_link": [],
                                        "user_id": user_id,
                                        "memory_type": "semantic"
                                    })
                        elif func_name == "no_operation":
                            decision.update({"reason": args.get("reason", "No reason provided"), "user_id": user_id})
                        
                        if decision["action"] != "NOOP" or "reason" in decision:
                            decisions.append(decision)
            except Exception as e:
                print(f"   ⚠️ Semantic memory management error: {e}")
        
        return decisions
    
    # --- Step 3: Decide (With ID Mapping) ---
    def step_decide(self, extract_result: Dict, context_bundles: List[Dict], user_id: str = 'default', training_mode: bool = False) -> List[Dict]:
        all_new_facts = extract_result['new_facts']
        
        # 1. 合并去重 Candidates
        temp_mem_storage = {}
        for bundle in context_bundles:
            for mem in bundle['candidates']:
                temp_mem_storage[mem['memory_id']] = mem
        
        unique_memories_list = list(temp_mem_storage.values())
        if not training_mode:
            print(f"🧠 [3. Manager] Global Decide: {len(all_new_facts)} facts vs {len(unique_memories_list)} memories.")

        # 🌟 2. 构造 ID 映射 (Mapping Logic)
        uuid_mapping = {}  # { "0": "real-uuid", "1": "real-uuid" }
        candidates_str = ""

        if not unique_memories_list:
            candidates_str = "(No relevant memories found. Treat as new topic.)"
        else:
            for idx, mem in enumerate(unique_memories_list):
                simple_id = str(idx)
                real_uuid = mem['memory_id']
                uuid_mapping[simple_id] = real_uuid
                candidates_str += f"[Memory Item ID: {simple_id}]\n- Content: {mem['content']}\n"
                
                # 添加关联的facts
                related_facts = mem.get('related_facts', [])
                if related_facts:
                    candidates_str += "- Related Facts:\n"
                    for fact_idx, fact in enumerate(related_facts):
                        candidates_str += f"  - Fact {fact_idx + 1}: {fact['text']}\n"
                        # 添加fact的details
                        details = fact.get('details', [])
                        if details:
                            if isinstance(details, list):
                                for detail in details:
                                    if isinstance(detail, dict):
                                        detail_str = ", ".join([f"{k}: {v}" for k, v in detail.items()])
                                        candidates_str += f"    Detail: {detail_str}\n"
                                    else:
                                        candidates_str += f"    Detail: {detail}\n"
                            elif isinstance(details, dict):
                                detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
                                candidates_str += f"    Detail: {detail_str}\n"
                candidates_str += "\n"

        # 构造最终 Prompt
        system_msg = MEMORY_MANAGER_PROMPT

        # 只提取事实的text字段，不包含details，避免LLM将details当作独立事实
        fact_texts = [fact['text'] for fact in all_new_facts]
        
        user_content = f"""
        [New Candidate Memories]
        {json.dumps(fact_texts, ensure_ascii=False)}
        
        [EXISTING MEMORIES]
        {candidates_str}
        """

        all_decisions = []
        try:
            # 直接调用非流式 API，无需思考过程
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                tools=MEMORY_TOOLS,
                tool_choice="required",
                temperature=0,
                stream=False
            )
            
            # 检查响应结构是否完整
            if not response.choices or len(response.choices) == 0:
                if not training_mode:
                    print(f"   ⚠️ Warning: No choices in response")
                return []
            
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                if not training_mode:
                    print(f"   ⚠️ Warning: No message in choice")
                return []
            
            tool_calls = choice.message.tool_calls
            if not tool_calls: return []

            # 🌟 辅助函数: 还原 ID
            def resolve_id(simple_id):
                real = uuid_mapping.get(str(simple_id))
                if not real and not training_mode:
                    print(f"   ⚠️ Warning: LLM hallucinated ID '{simple_id}', ignoring.")
                return real

            for tool_call in tool_calls:
                try:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    if not training_mode:
                        print(f"   🤖 Raw Action: {func_name} | Args: {args}")
                    decision = {"action": "NOOP"}

                    if func_name == "create_memory":
                        decision.update({
                            "action": "ADD", 
                            "summary": args.get("content", ""), 
                            "facts_to_link": args.get("evidence_facts", []),
                            "user_id": user_id
                        })
                    
                    elif func_name == "update_memory":
                        if "target_memory_id" in args:
                            real_tid = resolve_id(args["target_memory_id"])
                            if real_tid:
                                orig_created = temp_mem_storage.get(real_tid, {}).get('created_at', int(time.time()))
                                decision.update({
                                    "action": "UPDATE", 
                                    "target_id": real_tid, 
                                    "new_content": args.get("new_content", ""), 
                                    "facts_to_link": args.get("evidence_facts", []), 
                                    "orig_created": orig_created,
                                    "user_id": user_id
                                })

                    elif func_name == "delete_memory":
                        if "target_memory_id" in args:
                            real_tid = resolve_id(args["target_memory_id"])
                            if real_tid:
                                orig_created = temp_mem_storage.get(real_tid, {}).get('created_at', int(time.time()))
                                decision.update({
                                    "action": "DELETE", 
                                    "target_id": real_tid, 
                                    "facts_to_link": args.get("evidence_facts", []), 
                                    "orig_created": orig_created,
                                    "user_id": user_id
                                })

                    elif func_name == "infer_memory":
                        if "source_memory_ids" in args:
                            source_simples = args["source_memory_ids"]
                            # 确保source_simples是列表
                            if not isinstance(source_simples, list):
                                source_simples = [source_simples]
                            real_source_ids = [resolve_id(sid) for sid in source_simples if resolve_id(sid)]
                            if real_source_ids:
                                decision.update({
                                    "action": "INFER", 
                                    "source_ids": real_source_ids, 
                                    "summary": args.get("inference_content", ""), 
                                    "facts_to_link": args.get("evidence_facts", []),
                                    "user_id": user_id
                                })

                    elif func_name == "no_operation":
                        decision.update({"reason": args.get("reason", "No reason provided"), "user_id": user_id})
                    
                    if decision["action"] != "NOOP" or "reason" in decision:
                        all_decisions.append(decision)
                except Exception as e:
                    if not training_mode:
                        print(f"   ⚠️ Error processing tool call: {e}")
                    continue

        except Exception as e:
            if not training_mode:
                print(f"   ⚠️ Decision Error: {e}")
        
        return all_decisions
        
    # --- Batch Processing for Training with GRPO Support ---
    def batch_process(self, batch_data: List[Dict], user_id: str = 'default', grpo_compatible: bool = True) -> List[Dict]:
        """
        Batch processing for memory management training with GRPO compatibility.
        
        Args:
            batch_data (List[Dict]): List of input data for batch processing.
            user_id (str, optional): User ID for memory operations. Defaults to 'default'.
            grpo_compatible (bool, optional): Whether to return GRPO-compatible format. Defaults to True.
            
        Returns:
            List[Dict]: List of results for each input in the batch.
        """
        results = []
        
        for data in batch_data:
            # Extract facts from input text
            extract_result = self.step_extract(data['text'], extract_mode='whole')
            
            # Retrieve relevant memories
            context_bundles = self.step_retrieve(extract_result, limit=3, user_id=user_id)
            
            # Make decisions (memory operations) in training mode
            decisions = self.step_decide(extract_result, context_bundles, user_id=user_id, training_mode=True)
            
            # Execute decisions
            self.step_execute(decisions, extract_result, user_id=user_id)
            
            if grpo_compatible:
                # Format result for GRPO training
                result = {
                    'input': data['text'],
                    'extract_result': extract_result,
                    'decisions': decisions,
                    # Add GRPO-specific fields
                    'memory_operations': [d['action'] for d in decisions if d['action'] != 'NOOP'],
                    'memory_contents': [d.get('summary', '') for d in decisions if d['action'] != 'NOOP'],
                    # Ensure we have the expected_operation if provided in data
                    'expected_operation': data.get('expected_operation', '')
                }
            else:
                # Standard format for non-GRPO training
                result = {
                    'input': data['text'],
                    'extract_result': extract_result,
                    'decisions': decisions
                }
            
            results.append(result)
        
        return results

    # ==========================================
    # Step 4: Execute (Modified for Fact Inheritance)
    # ==========================================
    def step_execute(self, decisions: List[Dict], extract_result: Dict, user_id: str = 'default'):
        # 使用extract_result中的timestamp，而不是当前时间
        ts = extract_result['timestamp']
        chunk_id = extract_result['chunk_id']
        all_new_facts = extract_result['new_facts']
        
        # 1. 保存原始 Chunk
        # self.client.insert(self.chunk_col, [{"chunk_id": chunk_id, "text": extract_result["chunk_text"], "timestamp": ts, "embedding": get_embedding(extract_result["chunk_text"])}])

        # 2. 收集所有要链接的事实文本
        all_facts_to_link = set()
        for decision in decisions:
            action = decision.get("action")
            facts_to_link = decision.get('facts_to_link', [])
            for fact_text in facts_to_link:
                all_facts_to_link.add(fact_text)
        
        # 3. 对所有要处理的事实进行最终去重
        # 收集所有新事实
        all_facts = []
        for fact in all_new_facts:
            # 只处理在all_facts_to_link中的事实
            if fact['text'] in all_facts_to_link:
                all_facts.append(fact)
        
        # 对所有事实进行去重
        unique_all_facts = []
        seen_fact_keys = set()
        for fact in all_facts:
            fact_key = f"{fact['text']}"
            # 也考虑去掉"User"前缀的情况
            # stripped_fact_key = f"{fact['text'].lower().replace('user ', '')}"
            stripped_fact_key = fact_key
            if fact_key not in seen_fact_keys and stripped_fact_key not in seen_fact_keys:
                seen_fact_keys.add(fact_key)
                seen_fact_keys.add(stripped_fact_key)
                unique_all_facts.append(fact)
        
        if len(unique_all_facts) < len(all_facts):
            print(f"   ✅ 最终去重 {len(all_facts) - len(unique_all_facts)} 个重复事实")
        
        # 更新all_facts_to_link为去重后的事实文本集合
        all_facts_to_link = {fact['text'] for fact in unique_all_facts}

        # 3. 处理每个决策
        has_non_noop_action = False
        
        # 收集所有要链接的事实，确保去重
        all_matched_facts = []
        seen_fact_keys = set()
        
        for decision in decisions:
            action = decision.get("action")
            if action == "NOOP":
                self.operation_counts["NOOP"] += 1
                print(f"   🚫 No operation: {decision.get('reason', 'No reason provided')}")
                continue

            has_non_noop_action = True
            target_mem_id = None
            relations = []

            # --- CASE 1: ADD ---
            if action == "ADD":
                self.operation_counts["ADD"] += 1
                target_mem_id = str(uuid.uuid4())
                memory_type = decision.get('memory_type', 'semantic')
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", [], decision.get('user_id', 'default'), memory_type)
                print(f"   ✅ Created {memory_type} Mem: {target_mem_id[:8]}... | Content: {decision['summary']}")

            # --- CASE 2: UPDATE ---
            elif action == "UPDATE":
                self.operation_counts["UPDATE"] += 1
                target_mem_id = decision['target_id']
                memory_type = decision.get('memory_type', 'semantic')
                collection_name = self.episodic_col if memory_type == 'episodic' else self.semantic_col
                
                # 查询旧的memory内容
                old_memories = self.client.query(
                    collection_name=collection_name,
                    filter=f"memory_id == '{target_mem_id}'",
                    output_fields=["content", "created_at"]
                )
                
                old_content = "" if not old_memories else old_memories[0].get("content", "")
                new_content = decision['new_content']
                
                # 记录update前后的内容
                print(f"   🔄 Updating {memory_type} Mem: {target_mem_id[:8]}...")
                print(f"      Before: {old_content[:]}...")
                print(f"      After:  {new_content[:]}...")
                
                self._upsert_mem(target_mem_id, new_content, decision['orig_created'], ts, "active", [], decision.get('user_id', 'default'), memory_type)

            # --- CASE 3: DELETE ---
            elif action == "DELETE":
                self.operation_counts["DELETE"] += 1
                target_mem_id = decision['target_id']
                memory_type = decision.get('memory_type', 'semantic')
                self._upsert_mem(target_mem_id, "(Archived)", decision['orig_created'], ts, "archived", [], decision.get('user_id', 'default'), memory_type)
                print(f"   ❌ Deleted {memory_type} Mem: {target_mem_id[:8]}...")

            # --- CASE 4: INFER (With Fact Inheritance) ---
            elif action == "INFER":
                self.operation_counts["INFER"] += 1
                target_mem_id = str(uuid.uuid4()) # 这是 Memory C
                source_ids = decision.get('source_ids', []) # 这是 [A, B]
                memory_type = decision.get('memory_type', 'semantic')
                #############################################################
                # 查询source_ids对应的memory内容，用于打印
                source_mems = []
                if source_ids:
                    quoted_source_ids = [f'"{sid}"' for sid in source_ids]
                    mem_filter = f"status == 'active' and memory_id in [{','.join(quoted_source_ids)}]"
                    try:
                        # 查询所有可能的集合
                        collection_name = self.episodic_col if memory_type == 'episodic' else self.semantic_col
                        source_mems = self.client.query(
                            collection_name=collection_name,
                            filter=mem_filter,
                            output_fields=["content", "memory_id", "created_at", "user_id"]
                        )
                    except Exception as e:
                        print(f"   ⚠️ 查询source memory失败: {e}")
                #############################################################
                # 4.1 创建新记忆 C，并记录血缘关系 (inferred_from)
                relations = [{"type": "inferred_from", "target_id": sid} for sid in source_ids]
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", relations, decision.get('user_id', 'default'), memory_type)
                ####################################################################################
                # 将infer前后的memory内容拼在同一个字符串里输出
                infer_output = f"   💡 Inferred Mem: {target_mem_id[:8]}... | From: {[s[:8] for s in source_ids]}\n"
                infer_output += f"   ┌─────────────────────────────────────────────────────────────────────────────────\n"
                
                # 拼接infer前的memory内容
                if source_mems:
                    infer_output += f"   │ 📋 Infer 前的 Memory ({len(source_mems)}个):\n"
                    for mem in source_mems:
                        mem_id = mem.get("memory_id", "unknown")
                        content = mem.get("content", "")
                        infer_output += f"   │      📌 ID: {mem_id[:8]}... | 内容: {content[:]}...\n"
                
                # 拼接infer生成的memory内容
                infer_output += f"   │ 📝 Infer生成的 Memory:\n"
                infer_output += f"   │      📌 ID: {target_mem_id[:8]}... | 内容: {decision['summary'][:]}...\n"
                infer_output += f"   └─────────────────────────────────────────────────────────────────────────────────"
                
                # 一次性输出整个字符串
                print(infer_output)
                #################################################################################
                # 简化处理：不再从fact_col继承事实，而是直接在记忆中处理关系

            # --- CASE 5: TRAJECTORIZE (Similar to INFER) ---
            elif action == "TRAJECTORIZE":
                self.operation_counts["TRAJECTORIZE"] += 1
                target_mem_id = str(uuid.uuid4())
                source_ids = decision.get('source_ids', [])
                memory_type = decision.get('memory_type', 'semantic')
                
                # 查询source_ids对应的memory内容，用于打印
                source_mems = []
                if source_ids:
                    quoted_source_ids = [f'"{sid}"' for sid in source_ids]
                    mem_filter = f"status == 'active' and memory_id in [{','.join(quoted_source_ids)}]"
                    try:
                        collection_name = self.episodic_col if memory_type == 'episodic' else self.semantic_col
                        source_mems = self.client.query(
                            collection_name=collection_name,
                            filter=mem_filter,
                            output_fields=["content", "memory_id", "created_at", "user_id"]
                        )
                    except Exception as e:
                        print(f"   ⚠️ 查询source memory失败: {e}")
                
                # 创建新记忆，并记录血缘关系 (trajectorized_from)
                relations = [{"type": "trajectorized_from", "target_id": sid} for sid in source_ids]
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", relations, decision.get('user_id', 'default'), memory_type)
                
                # 打印日志
                traj_output = f"   📈 Trajectorized Mem: {target_mem_id[:8]}... | From: {[s[:8] for s in source_ids]}\n"
                traj_output += f"   ┌─────────────────────────────────────────────────────────────────────────────────\n"
                
                if source_mems:
                    traj_output += f"   │ 📋 Trajectorize 前的 Memory ({len(source_mems)}个):\n"
                    for mem in source_mems:
                        mem_id = mem.get("memory_id", "unknown")
                        content = mem.get("content", "")
                        traj_output += f"   │      📌 ID: {mem_id[:8]}... | 内容: {content[:]}...\n"
                
                traj_output += f"   │ 📝 Trajectorize 生成的 Memory:\n"
                traj_output += f"   │      📌 ID: {target_mem_id[:8]}... | 内容: {decision['summary'][:]}...\n"
                traj_output += f"   └─────────────────────────────────────────────────────────────────────────────────"
                
                print(traj_output)

            # --- Common: Link NEW Facts for this decision ---
            # 无论是 ADD, UPDATE 还是 INFER，都会把当前决策的新证据关联上去
            facts_to_link = decision.get('facts_to_link', [])
            if target_mem_id and facts_to_link:
                # 查找与待链接事实文本匹配的完整事实对象（包含details和fact_id）
                for fact_text in facts_to_link:
                    # 在所有新事实中查找匹配的文本，以获取完整的fact对象（包含details和fact_id）
                    matching_fact = next((f for f in all_new_facts if f['text'] == fact_text), None)
                    if matching_fact:
                        # 检查事实是否已经被处理过
                        fact_key = f"{matching_fact['text']}"
                        # 也考虑去掉"User"前缀的情况
                        # stripped_fact_key = f"{matching_fact['text'].lower().replace('user ', '')}"
                        stripped_fact_key = fact_key
                        if fact_key not in seen_fact_keys and stripped_fact_key not in seen_fact_keys:
                            seen_fact_keys.add(fact_key)
                            seen_fact_keys.add(stripped_fact_key)
                            # 添加目标记忆ID到事实中
                            fact_with_target = matching_fact.copy()
                            fact_with_target['target_mem_id'] = target_mem_id
                            all_matched_facts.append(fact_with_target)
                    else:
                        # 如果没有找到匹配的完整事实对象，使用文本创建一个简单的事实对象
                        new_fact = {'text': fact_text, 'fact_id': str(uuid.uuid4()), 'target_mem_id': target_mem_id}
                        fact_key = f"{new_fact['text']}"
                        # stripped_fact_key = f"{new_fact['text'].lower().replace('user ', '')}"
                        stripped_fact_key = fact_key
                        if fact_key not in seen_fact_keys and stripped_fact_key not in seen_fact_keys:
                            seen_fact_keys.add(fact_key)
                            seen_fact_keys.add(stripped_fact_key)
                            all_matched_facts.append(new_fact)
        
        # 简化处理：不再存储事实到单独的集合，而是将事实信息直接包含在记忆中
        if all_matched_facts:
            print(f"   🔗 关联 {len(all_matched_facts)} 个事实到对应记忆")

    def _upsert_mem(self, mem_id, content, c_at, u_at, status, relations, user_id, memory_type='semantic'):
        """
        插入或更新记忆
        
        Args:
            mem_id: 记忆ID
            content: 记忆内容
            c_at: 创建时间戳
            u_at: 更新时间戳
            status: 状态
            relations: 关系列表
            user_id: 用户标识
            memory_type: 记忆类型，可选值：semantic, episodic
        """
        collection_name = self.episodic_col if memory_type == 'episodic' else self.semantic_col
        
        # 构建基础数据
        data = {
            "memory_id": mem_id,
            "embedding": get_embedding(content),
            "content": content,
            "user_id": user_id,
            "status": status,
            "created_at": c_at,
            "updated_at": u_at,
            "relations": relations
        }
        
        # 对于episodic memory，尝试从内容中提取日期
        if memory_type == 'episodic':
            # 尝试解析格式："YYYY-MM-DD: Summary | Details: Detailed description"
            if ": " in content:
                date_part = content.split(": ", 1)[0]
                # 检查是否是有效的日期格式
                if len(date_part) == 10 and date_part[4] == '-' and date_part[7] == '-':
                    try:
                        # 验证日期格式
                        from datetime import datetime
                        datetime.strptime(date_part, "%Y-%m-%d")
                        # 如果是有效的日期，添加到数据中
                        data["date"] = date_part
                    except ValueError:
                        # 不是有效的日期，忽略
                        pass
        
        self.client.upsert(collection_name, [data])

    def step_preprocess_facts(self, extract_result: Dict, user_id: str = 'default') -> Dict:
        """
        预处理提取出的事实，确保从源头上去重
        
        Args:
            extract_result: 提取结果字典，包含new_facts
            user_id: 用户标识，确保只处理当前用户的事实
            
        Returns:
            更新后的提取结果字典，包含fact_id信息
        """
        new_facts = extract_result['new_facts']
        processed_facts = []
        
        print(f"🔍 [Preprocess Facts] 检查 {len(new_facts)} 个事实是否已存在...")
        
        # 对同一批次内的事实进行去重，避免同一批次中重复的事实被处理
        unique_facts_in_batch = []
        seen_fact_keys = set()
        for fact in new_facts:
            # 使用fact_text作为唯一标识
            fact_key = f"{fact['text']}"
            if fact_key not in seen_fact_keys:
                seen_fact_keys.add(fact_key)
                unique_facts_in_batch.append(fact)
        
        if len(unique_facts_in_batch) < len(new_facts):
            print(f"   ✅ 同一批次内去重 {len(new_facts) - len(unique_facts_in_batch)} 个重复事实")
        
        for fact in unique_facts_in_batch:
            fact_text = fact['text']
            
            # 为每个事实生成唯一ID
            fact_id = str(uuid.uuid4())
            
            # 构建处理后的事实
            processed_fact = {
                "text": fact_text,
                "fact_id": fact_id,
                "memory_type": fact.get('memory_type', 'semantic')
            }
            
            processed_facts.append(processed_fact)
        
        # 更新提取结果
        extract_result['new_facts'] = processed_facts
        return extract_result
    
    def process(self, text, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None, timestamp: int = None, max_history_turns: int = 5):
        # 1. 提取记忆
        res = self.step_extract(text, extract_mode=extract_mode, timestamp=timestamp, max_history_turns=max_history_turns)
        if not res['new_facts']: return
        
        # 2. 预处理事实，检查是否已存在
        res = self.step_preprocess_facts(res, user_id=user_id)
        
        # 3. 检查预处理后是否还有新事实
        if not res['new_facts']:
            print(f"   ✅ 所有事实都已存在，无需处理")
            return
        
        print(f"   新证据: {res['new_facts']}")
        
        # 4. 获取提取的episodic和semantic memory
        episodic_memories = res.get('episodic_memories', [])
        semantic_memories = res.get('semantic_memories', [])
        
        # 5. 分别检索相关的旧记忆
        # 检索episodic memory
        episodic_retrieved = []
        if episodic_memories:
            # 构建episodic memory的检索结果
            for mem in episodic_memories:
                facts = mem.get('facts', [])
                for fact in facts:
                    fact['memory_type'] = 'episodic'
            # 使用step_retrieve检索相关记忆
            episodic_ctx = self.step_retrieve(res, limit=retrieve_limit, user_id=user_id, similarity_threshold=similarity_threshold)
            # 提取检索到的episodic memory
            for bundle in episodic_ctx:
                if bundle.get('memory_type') == 'episodic':
                    episodic_retrieved.extend(bundle.get('candidates', []))
        
        # 检索semantic memory
        semantic_retrieved = []
        if semantic_memories:
            # 构建semantic memory的检索结果
            for mem in semantic_memories:
                facts = mem.get('facts', [])
                for fact in facts:
                    fact['memory_type'] = 'semantic'
            # 使用step_retrieve检索相关记忆
            semantic_ctx = self.step_retrieve(res, limit=retrieve_limit, user_id=user_id, similarity_threshold=similarity_threshold)
            # 提取检索到的semantic memory
            for bundle in semantic_ctx:
                if bundle.get('memory_type') == 'semantic':
                    semantic_retrieved.extend(bundle.get('candidates', []))
        
        # 6. 分别管理episodic和semantic memory
        episodic_decisions = []
        if episodic_memories:
            episodic_decisions = self.step_manage_episodic_memory(episodic_memories, episodic_retrieved, user_id=user_id)
        
        semantic_decisions = []
        if semantic_memories:
            semantic_decisions = self.step_manage_semantic_memory(semantic_memories, semantic_retrieved, user_id=user_id)
        
        # 7. 合并所有决策并执行
        all_decisions = []
        all_decisions.extend(episodic_decisions)
        all_decisions.extend(semantic_decisions)
        
        if all_decisions:
            self.step_execute(all_decisions, res, user_id=user_id)
        
        # 8. 管理Core Memory
        if episodic_memories or semantic_memories:
            self.step_manage_core_memory(episodic_memories, semantic_memories)
        
    def process_user_memory_infer(self, line, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None, max_history_turns: int = 5):
        """处理用户记忆会话，支持longmemeval数据集格式"""
        # 重置操作计数，确保每个用户的计数独立
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        dates = line.get("haystack_dates")
        sessions = line.get("haystack_sessions")

        for session_id, session in enumerate(sessions):
            date = dates[session_id] + " UTC"
            date_format = "%Y/%m/%d (%a) %H:%M UTC"
            date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
            # 生成timestamp
            timestamp = int(date_string.timestamp())
            
            print(f"处理会话 {session_id + 1}/{len(sessions)}: {dates[session_id]}")
            
            # 直接传递session对象给process方法，而不是转换为文本
            # 使用现有的process方法处理会话消息，传递user_id、similarity_threshold和timestamp
            self.process(session, retrieve_limit=retrieve_limit, extract_mode=extract_mode, user_id=user_id, similarity_threshold=similarity_threshold, timestamp=timestamp, max_history_turns=max_history_turns)
        
        # 返回操作次数统计
        return self.operation_counts
        
    def search_memories(self, query_text, top_k=5, user_id: str = 'default', threshold: float = 0.0, similarity_threshold: float = None, enhanced_search: bool = False, use_bm25: bool = True, bm25_weight: float = 0.5):
        """搜索记忆并返回相关结果
        
        Args:
            query_text: 查询文本
            top_k: 返回的记忆数量上限
            user_id: 用户标识，确保只检索当前用户的记忆
            threshold: 相似度阈值，低于该阈值的记忆将被过滤掉
            similarity_threshold: 向量数据库搜索时的相似度阈值，低于该阈值的记忆将被过滤掉
            enhanced_search: 是否启用增强型搜索模式，启用后会增强搜索逻辑
            use_bm25: 是否使用BM25检索
            bm25_weight: BM25检索结果的权重，范围0-1
        """
        query_vec = get_embedding(query_text)
        
        # 添加调试信息
        filter_expr = f"status == 'active' and user_id == '{user_id}'"
        print(f"   🔍 搜索过滤条件: {filter_expr}, 阈值: {threshold}, 向量搜索阈值: {similarity_threshold}, 使用BM25: {use_bm25}")
        
        # 搜索semantic memory集合
        semantic_res = self.client.search(
            self.semantic_col, [query_vec], filter=filter_expr, limit=top_k,  # 搜索更多记忆，避免遗漏
            output_fields=["content", "memory_id", "created_at", "user_id"],  # 包含user_id字段用于调试
            similarity_threshold=similarity_threshold
        )
        
        # 搜索episodic memory集合
        episodic_res = self.client.search(
            self.episodic_col, [query_vec], filter=filter_expr, limit=top_k,  # 搜索更多记忆，避免遗漏
            output_fields=["content", "memory_id", "created_at", "user_id"],  # 包含额外字段
            similarity_threshold=similarity_threshold
        )
        
        # 合并搜索结果
        all_memories = []
        
        # 处理semantic memory结果
        if semantic_res and semantic_res[0]:
            for hit in semantic_res[0]:
                memory = hit['entity']
                # 在Milvus中，使用余弦相似度时，distance表示1 - 相似度，范围[0, 2]
                # 转换为相似度得分，范围[0, 1]，值越大表示越相似
                similarity_score = max(0, 1 - hit['distance'])
                memory["original_score"] = similarity_score
                memory["vector_score"] = similarity_score
                memory["memory_type"] = "semantic"
                all_memories.append(memory)
        
        # 处理episodic memory结果
        if episodic_res and episodic_res[0]:
            for hit in episodic_res[0]:
                memory = hit['entity']
                # 在Milvus中，使用余弦相似度时，distance表示1 - 相似度，范围[0, 2]
                # 转换为相似度得分，范围[0, 1]，值越大表示越相似
                similarity_score = max(0, 1 - hit['distance'])
                memory["original_score"] = similarity_score
                memory["vector_score"] = similarity_score
                memory["memory_type"] = "episodic"
                all_memories.append(memory)
        
        # 如果启用BM25检索
        if use_bm25:
            try:
                # BM25检索semantic memory
                # 将文本匹配条件加入filter中
                bm25_filter = f"{filter_expr} and content like '%{query_text}%'"
                
                semantic_bm25_res = self.client.search(
                    self.semantic_col, [query_vec], filter=bm25_filter, limit=top_k,
                    output_fields=["content", "memory_id", "created_at", "user_id"]
                )
                
                # BM25检索episodic memory
                episodic_bm25_res = self.client.search(
                    self.episodic_col, [query_vec], filter=bm25_filter, limit=top_k,
                    output_fields=["content", "memory_id", "created_at", "user_id"]
                )
                
                # 处理BM25检索结果
                if semantic_bm25_res and semantic_bm25_res[0]:
                    for hit in semantic_bm25_res[0]:
                        memory = hit['entity']
                        # 在Milvus中，使用余弦相似度时，distance表示1 - 相似度，范围[0, 2]
                        # 转换为相似度得分，范围[0, 1]，值越大表示越相似
                        similarity_score = max(0, 1 - hit['distance'])
                        memory["bm25_score"] = similarity_score
                        memory["memory_type"] = "semantic"
                        # 检查是否已存在，不存在则添加
                        existing = next((m for m in all_memories if m.get("memory_id") == memory.get("memory_id")), None)
                        if not existing:
                            memory["original_score"] = similarity_score
                            memory["vector_score"] = similarity_score
                            all_memories.append(memory)
                        else:
                            # 如果已存在，更新BM25得分
                            existing["bm25_score"] = similarity_score
                
                if episodic_bm25_res and episodic_bm25_res[0]:
                    for hit in episodic_bm25_res[0]:
                        memory = hit['entity']
                        # 在Milvus中，使用余弦相似度时，distance表示1 - 相似度，范围[0, 2]
                        # 转换为相似度得分，范围[0, 1]，值越大表示越相似
                        similarity_score = max(0, 1 - hit['distance'])
                        memory["bm25_score"] = similarity_score
                        memory["memory_type"] = "episodic"
                        # 检查是否已存在，不存在则添加
                        existing = next((m for m in all_memories if m.get("memory_id") == memory.get("memory_id")), None)
                        if not existing:
                            memory["original_score"] = similarity_score
                            memory["vector_score"] = similarity_score
                            all_memories.append(memory)
                        else:
                            # 如果已存在，更新BM25得分
                            existing["bm25_score"] = similarity_score
            except Exception as bm25_error:
                # 忽略BM25检索错误
                print(f"BM25检索失败 (忽略): {bm25_error}")
        
        # 计算综合分数并排序
        results = []
        for memory in all_memories:
            vector_score = memory.get("vector_score", 0)
            bm25_score = memory.get("bm25_score", 0)
            
            # 计算综合分数
            if use_bm25 and bm25_score > 0:
                # 融合向量分数和BM25分数
                memory["combined_score"] = vector_score * (1 - bm25_weight) + bm25_score * bm25_weight
            else:
                # 仅使用向量分数
                memory["combined_score"] = vector_score
            
            # 根据阈值过滤结果
            if memory["combined_score"] >= threshold:
                results.append(memory)
        
        # 根据综合分数重新排序记忆
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        # 确保返回的记忆数量不超过top_k
        return results[:top_k]
        
    def _calculate_memory_score(self, memory, enhanced_search=False):
        """直接返回memory与query的内积，不考虑关联事实的相关性"""
        original_score = memory.get("original_score", 0)
        # 直接使用memory与query的内积作为综合分数
        memory["combined_score"] = original_score
        return memory
        
    def _generate_fact_embedding(self, text, details):
        """生成事实的embedding，将text和details拼接起来
        
        Args:
            text: 事实的文本
            details: 事实的详细信息，类型为列表
            
        Returns:
            生成的embedding向量
        """
        # 将details拼接成字符串
        details_str = ""
        if isinstance(details, list) and details:
            # 遍历details列表，将每个details项转换为字符串
            for i, detail in enumerate(details):
                if isinstance(detail, dict):
                    # 如果detail是字典，转换为键值对字符串
                    detail_str = ", ".join([f"{k}: {v}" for k, v in detail.items()])
                    details_str += f"Detail {i+1}: {detail_str}\n"
                else:
                    # 否则直接转换为字符串
                    details_str += f"Detail {i+1}: {str(detail)}\n"
        elif isinstance(details, dict):
            # 如果details是字典，转换为键值对字符串
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
        
        # 将text和details拼接成完整的文本
        if details_str:
            full_text = f"{text}\n\nDetails:\n{details_str.strip()}"
        else:
            full_text = text
        
        # 生成embedding
        return get_embedding(full_text)
        
    def generate_response(self, question, question_date, context):
        """生成问题响应"""
        prompt = LME_ANSWER_PROMPT.format(
            question=question,
            question_date=question_date,
            context=context
        )
        response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0,
                )
        
        return response

# ==========================================
# 评估相关函数
# ==========================================
def response_user(line, pipeline, retrieve_limit=30, max_facts_per_memory=3, user_id='default', threshold: float = 0.3, enhanced_search: bool = True, use_bm25: bool = True, bm25_weight: float = 0.5):
    """处理用户问题，生成响应
    
    Args:
        line: 包含问题和其他信息的字典
        pipeline: MemoryPipeline实例
        retrieve_limit: 检索记忆的数量限制
        max_facts_per_memory: 每个记忆的事实数量限制
        user_id: 用户标识，确保只检索当前用户的记忆
        threshold: 相似度阈值，低于该阈值的记忆将被过滤掉
        enhanced_search: 是否启用增强型搜索模式，启用后会调大topk并增强rerank
    """
    question = line.get("question")
    question_date = line.get("question_date")
    question_date = question_date + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    
    # 增强型搜索模式：调大topk
    if enhanced_search:
        # 调大初始检索数量，例如乘以2
        enhanced_top_k = retrieve_limit * 2
        print(f"   🚀 启用增强型搜索模式，初始检索数量: {enhanced_top_k}")
    else:
        enhanced_top_k = retrieve_limit
    
    # 搜索记忆，传递user_id、threshold、enhanced_search和BM25相关参数
    retrieved_memories = pipeline.search_memories(question, top_k=enhanced_top_k, user_id=user_id, threshold=threshold, enhanced_search=enhanced_search, use_bm25=use_bm25, bm25_weight=bm25_weight)
    
    # 确保retrieved_memories不是None
    retrieved_memories = retrieved_memories or []
    
    # 构建上下文，包含记忆和关联的事实
    memories_with_facts = []
    
    # 分别处理episodic和semantic memory
    episodic_memories = []
    semantic_memories = []
    
    for mem in retrieved_memories:
        memory_type = mem.get("memory_type", "semantic")
        if memory_type == "episodic":
            episodic_memories.append(mem)
        else:
            semantic_memories.append(mem)
    
    # 添加episodic memory
    if episodic_memories:
        memories_with_facts.append("## Episodic Memories")
        for mem in episodic_memories:
            memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
            memories_with_facts.append(memory_line)
    
    # 添加semantic memory
    if semantic_memories:
        memories_with_facts.append("## Semantic Memories")
        for mem in semantic_memories:
            memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
            memories_with_facts.append(memory_line)
    
    # 添加core memory
    if pipeline.core_memory:
        memories_with_facts.append("## Core Memory")
        memories_with_facts.append(f"{pipeline.core_memory}")
    
    memories_str = "\n".join(memories_with_facts)
    
    # 生成响应
    response = pipeline.generate_response(question, question_date_string, memories_str)
    answer = response.choices[0].message.content
    
    return retrieved_memories, answer

def process_and_evaluate_user(line, user_index, infer=True, retrieve_limit: int = 3, extract_mode: str = "whole", vector_db_type="milvus", dataset_name="", max_history_turns: int = 5, use_bm25: bool = True, bm25_weight: float = 0.5):
    """
    封装单个用户的所有处理步骤，以便并行执行。
    返回一个包含所有统计信息的字典。
    """
    try:
        # 为每个用户生成唯一的user_id，确保记忆隔离
        user_id = f"user_{user_index}"
        
        # 为每个用户创建独立的pipeline实例，避免多线程竞争
        # 注意：每个用户的pipeline实例不应该清空数据库，clear_db固定为False
        pipeline = MemoryPipeline(vector_db_type=vector_db_type, clear_db=False, dataset_name=dataset_name)
        
        # 处理用户记忆会话，传递user_id、extract_mode和max_history_turns
        memory_counts = pipeline.process_user_memory_infer(line, retrieve_limit=retrieve_limit, extract_mode=extract_mode, user_id=user_id, max_history_turns=max_history_turns)
        
        # 生成问题响应，传递user_id、BM25相关参数
        retrieved_memories, answer = response_user(line, pipeline, retrieve_limit=30, user_id=user_id, threshold=0.3, enhanced_search=True, use_bm25=True, bm25_weight=0.5)
        
        # 确保retrieved_memories不是None
        retrieved_memories = retrieved_memories or []
        
        # 构建上下文字符串用于后续处理
        memories_with_facts = []
        
        # 分别处理episodic和semantic memory
        episodic_memories = []
        semantic_memories = []
        
        for mem in retrieved_memories:
            memory_type = mem.get("memory_type", "semantic")
            if memory_type == "episodic":
                episodic_memories.append(mem)
            else:
                semantic_memories.append(mem)
        
        # 添加episodic memory
        if episodic_memories:
            memories_with_facts.append("## Episodic Memories")
            for mem in episodic_memories:
                memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
                memories_with_facts.append(memory_line)
        
        # 添加semantic memory
        if semantic_memories:
            memories_with_facts.append("## Semantic Memories")
            for mem in semantic_memories:
                memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
                memories_with_facts.append(memory_line)
        
        # 添加core memory
        if pipeline.core_memory:
            memories_with_facts.append("## Core Memory")
            memories_with_facts.append(f"{pipeline.core_memory}")
        
        memories_str = "\n".join(memories_with_facts)
        
        # 获取标准答案和问题类型
        golden_answer = line.get("answer")
        question = line.get("question")
        question_type = line.get("question_type", "unknown")
        
        # 评估答案正确性
        is_correct = lme_grader(llm_client, question, golden_answer, answer)
        
        return {
            "index": user_index,
            "is_correct": is_correct,
            "counts": memory_counts,
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "golden_answer": golden_answer,
            "retrieved_memories": retrieved_memories,
            "context": memories_str,
        }
    except Exception as e:
        print(f"处理用户 {user_index} 出错 ({line.get('question', 'Unknown')[:20]}...): {e}")
        return {
            "index": user_index,
            "is_correct": False,
            "counts": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0},
            "question": line.get("question", "N/A"),
            "question_type": line.get("question_type", "unknown"),
            "context": "N/A",
            "answer": "N/A",
            "golden_answer": line.get("answer", "N/A"),
            "retrieved_memories": []
        }

# ==========================================
# Main Test & Evaluation
# ==========================================
if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Memory Pipeline with longmemeval Evaluation")
    parser.add_argument("--eval", action="store_true", help="是否进行评估")
    parser.add_argument("--infer", action="store_true", default=True, help="是否使用推理功能")
    parser.add_argument("--num_users", type=int, default=50, help="评估用户数量")
    parser.add_argument("--max_workers", type=int, default=10, help="并行处理的工作线程数")
    parser.add_argument("--retrieve_limit", type=int, default=3, help="检索时返回的记忆数量")
    parser.add_argument("--threshold", type=float, default=0.7, help="记忆相似度阈值，低于该阈值的记忆将被过滤掉")
    parser.add_argument("--extract-mode", type=str, default="whole", choices=["whole", "turn"], help="提取模式：whole-对整个chunk进行提取，turn-按轮次提取，包含chat history")
    parser.add_argument("--max-history-turns", type=int, default=5, help="当extract-mode为turn时，使用的聊天历史轮数")
    parser.add_argument("--vector-db-type", type=str, default="milvus", choices=["milvus", "qdrant"], help="指定使用的向量数据库类型")
    parser.add_argument("--clear-db", action="store_true", help="运行前清空数据库")
    parser.add_argument("--data-path", type=str, help="指定数据文件路径")
    parser.add_argument("--dataset-type", type=str, default="longmemeval", choices=["longmemeval", "hotpotqa"], help="指定数据集类型")
    parser.add_argument("--use-bm25", action="store_true", default=True, help="是否使用BM25检索")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="BM25检索结果的权重，范围0-1")
    args = parser.parse_args()
    
    # 初始化内存管道
    pipeline = MemoryPipeline(vector_db_type=args.vector_db_type, clear_db=args.clear_db, mode='eval' if args.eval else 'test', dataset_name=args.dataset_type)
    
    if args.eval:
        # 评估模式
        try:
            # 根据数据集类型设置默认数据路径
            if args.dataset_type == "hotpotqa":
                data_path = args.data_path or "./data/hotpotqa-val.jsonl"
            else:  # longmemeval
                data_path = args.data_path or "./data/longmemeval_s_cleaned.json"
                
            print(f"调试信息：")
            print(f"  数据集类型：{args.dataset_type}")
            print(f"  指定的数据路径：{args.data_path}")
            print(f"  实际使用的数据路径：{data_path}")
            print(f"  文件是否存在：{os.path.exists(data_path)}")
            
            if not os.path.exists(data_path):
                print(f"数据集文件不存在: {data_path}")
                exit()
            
            # 判断文件类型并加载数据
            lines = []
            print(f"  文件格式：{'JSONL' if data_path.endswith('.jsonl') else 'JSON'}")
            if data_path.endswith(".jsonl"):
                # 处理JSONL格式文件
                print(f"  开始加载JSONL文件...")
                with open(data_path, "r") as f:
                    for i, line in enumerate(f):
                        lines.append(json.loads(line.strip()))
                        if i < 2:  # 打印前2条数据的关键字段
                            loaded_item = lines[-1]
                            print(f"    第{i+1}条数据关键字段：")
                            print(f"      是否包含context：{'context' in loaded_item}")
                            print(f"      是否包含haystack_dates：{'haystack_dates' in loaded_item}")
                            print(f"      数据ID：{loaded_item.get('id', '未知')}")
            else:
                # 处理JSON格式文件
                print(f"  开始加载JSON文件...")
                with open(data_path, "r") as f:
                    lines = json.load(f)
                    if lines and len(lines) > 0:
                        print(f"    共加载 {len(lines)} 条数据")
                        if len(lines) > 0:
                            loaded_item = lines[0]
                            print(f"    第1条数据关键字段：")
                            print(f"      是否包含context：{'context' in loaded_item}")
                            print(f"      是否包含haystack_dates：{'haystack_dates' in loaded_item}")
                            print(f"      数据ID：{loaded_item.get('id', '未知')}")
                    
            # 如果num_users为-1，加载所有数据；否则加载指定数量
            if args.num_users != -1:
                lines = lines[:args.num_users]
            
            # 数据集格式判断和转换
            def is_valid_format(line):
                """判断数据条目是否符合longmemeval格式要求"""
                return "haystack_dates" in line and "haystack_sessions" in line
            
            def convert_hotpotqa_to_expected_format(hotpotqa_item):
                """将HotpotQA条目转换为预期格式"""
                # 生成固定格式的日期
                date = "2023/01/01 (Sun) 12:00"
                
                # 构建系统消息，包含所有背景知识
                context = hotpotqa_item["context"]
                system_content = "以下是背景知识：\n"
                for title, sentences in zip(context["title"], context["sentences"]):
                    system_content += f"\n{title}:\n"
                    for sentence in sentences:
                        system_content += f"- {sentence}\n"
                
                # 构建用户消息，包含问题
                user_content = hotpotqa_item["question"]
                
                # 构建会话结构
                session = [
                    {"role": "system", "content": system_content.strip()},
                    {"role": "user", "content": user_content}
                ]
                
                # 返回转换后的格式，包含question_type字段
                return {
                    "haystack_dates": [date],
                    "haystack_sessions": [session],
                    "id": hotpotqa_item["id"],
                    "answer": hotpotqa_item["answer"],
                    "question_type": hotpotqa_item.get("type", "unknown")  # 使用hotpotqa的type字段作为question_type
                }
            
            # 如果是hotpotqa数据集，检查格式并转换
            if args.dataset_type == "hotpotqa":
                # 检查第一个条目是否符合格式要求
                if lines and not is_valid_format(lines[0]):
                    print(f"HotpotQA数据集格式不符合要求，正在转换 {len(lines)} 个条目...")
                    # 转换所有条目
                    converted_lines = []
                    for i, item in enumerate(lines):
                        if i % 100 == 0:  # 每处理100条打印一次进度
                            print(f"已转换 {i}/{len(lines)} 条数据")
                        converted_lines.append(convert_hotpotqa_to_expected_format(item))
                    lines = converted_lines
                    print(f"转换完成，共转换 {len(lines)} 个条目")
                else:
                    print("HotpotQA数据集格式符合要求，正在检查并确保所有条目包含question_type字段...")
                    # 确保所有条目都包含question_type字段
                    for i, line in enumerate(lines):
                        if "question_type" not in line:
                            # 尝试从原始数据中获取type字段，如果没有则使用默认值
                            lines[i]["question_type"] = line.get("type", "unknown")
                    print("检查完成，所有条目都包含question_type字段")
            
            print(f"已加载 {len(lines)} 个用户/问题。")

            user_detail_results = []
            total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
            
            # 并行处理用户
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交任务 - 确保参数顺序正确：line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type, args.max_history_turns
                # 注意：这里clear_db固定为False，只在主函数中执行一次清空操作
                future_to_user = {executor.submit(process_and_evaluate_user, line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type, args.max_history_turns, args.use_bm25, args.bm25_weight): (line, idx) for idx, line in enumerate(lines)}
                
                # 处理结果
                for future in tqdm(as_completed(future_to_user), total=len(future_to_user)):
                    line, idx = future_to_user[future]
                    try:
                        result = future.result()
                        user_detail_results.append(result)
                        
                        # 统计总操作次数
                        for key, value in result["counts"].items():
                            total_memory_counts[key] += value
                    except Exception as e:
                        print(f"处理用户 {idx} 时发生错误: {e}")
            
            # 计算总准确率
            correct_count = sum(1 for result in user_detail_results if result["is_correct"])
            accuracy = correct_count / len(user_detail_results) * 100 if user_detail_results else 0
            
            # 按question_type统计每类问题的准确率
            question_type_stats = {}
            for result in user_detail_results:
                q_type = result.get("question_type", "unknown")
                if q_type not in question_type_stats:
                    question_type_stats[q_type] = {"total": 0, "correct": 0}
                question_type_stats[q_type]["total"] += 1
                if result["is_correct"]:
                    question_type_stats[q_type]["correct"] += 1
            
            # 输出评估结果
            print("\n" + "="*50)
            print(f"{args.dataset_type} 评估结果")
            print("="*50)
            print(f"总用户数: {len(user_detail_results)}")
            print(f"正确回答数: {correct_count}")
            print(f"总准确率: {accuracy:.2f}%")
            print(f"记忆操作总数:")
            for op, count in total_memory_counts.items():
                print(f"  {op}: {count}")
            
            # 输出按question_type分类的准确率
            print("\n" + "="*50)
            print("按问题类型分类的准确率")
            print("="*50)
            for q_type, stats in question_type_stats.items():
                type_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"{q_type}: {stats['correct']}/{stats['total']} ({type_accuracy:.2f}%)")
            
            print("="*50)
            
            # 输出详细结果
            print("\n详细结果:")
            for result in user_detail_results:  
                print(f"用户 {result['index']}: {'✓' if result['is_correct'] else '✗'}")
                print(f"  问题: {result['question']}")
                print(f"  问题类型: {result.get('question_type', 'unknown')}")
                print(f"  上下文: {result['context']}")
                print(f"  回答: {result['answer']}...")
                print(f"  标准答案: {result['golden_answer']}...")
                print(f"  记忆操作: {result['counts']}")
                print()
                
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        try:
            # 测试模式
            # 根据数据集类型设置默认数据路径
            if args.dataset_type == "hotpotqa":
                data_path = args.data_path or "./data/hotpotqa-val.jsonl"
            else:  # longmemeval
                data_path = args.data_path or "./data/longmemeval_s_cleaned_test.json"
                
            if not os.path.exists(data_path):
                print(f"数据集文件不存在: {data_path}")
                exit()
            
            # 判断文件类型并加载数据
            lines = []
            if data_path.endswith(".jsonl"):
                # 处理JSONL格式文件
                with open(data_path, "r") as f:
                    for line in f:
                        lines.append(json.loads(line.strip()))
            else:
                # 处理JSON格式文件
                with open(data_path, "r") as f:
                    lines = json.load(f)
                    
            if args.num_users != -1:
                lines = lines[:args.num_users]
            
            print(f"已加载 {len(lines)} 个用户/问题。")

            user_detail_results = []
            total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
            
            # 并行处理用户
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交任务，包含extract_mode和dataset_type参数
                future_to_user = {executor.submit(process_and_evaluate_user, line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type, args.max_history_turns, args.use_bm25, args.bm25_weight): (line, idx) for idx, line in enumerate(lines)}
                
                # 处理结果
                for future in tqdm(as_completed(future_to_user), total=len(future_to_user)):
                    line, idx = future_to_user[future]
                    try:
                        result = future.result()
                        user_detail_results.append(result)
                        
                        # 统计总操作次数
                        for key, value in result["counts"].items():
                            total_memory_counts[key] += value
                    except Exception as e:
                        print(f"处理用户 {idx} 时发生错误: {e}")
            
            # 计算总准确率
            correct_count = sum(1 for result in user_detail_results if result["is_correct"])
            accuracy = correct_count / len(user_detail_results) * 100 if user_detail_results else 0
            
            # 按question_type统计每类问题的准确率
            question_type_stats = {}
            for result in user_detail_results:
                q_type = result.get("question_type", "unknown")
                if q_type not in question_type_stats:
                    question_type_stats[q_type] = {"total": 0, "correct": 0}
                question_type_stats[q_type]["total"] += 1
                if result["is_correct"]:
                    question_type_stats[q_type]["correct"] += 1
            
            # 输出评估结果
            print("\n" + "="*50)
            print("LongMemEval 评估结果")
            print("="*50)
            print(f"总用户数: {len(user_detail_results)}")
            print(f"正确回答数: {correct_count}")
            print(f"总准确率: {accuracy:.2f}%")
            print(f"记忆操作总数:")
            for op, count in total_memory_counts.items():
                print(f"  {op}: {count}")
            
            # 输出按question_type分类的准确率
            print("\n" + "="*50)
            print("按问题类型分类的准确率")
            print("="*50)
            for q_type, stats in question_type_stats.items():
                type_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"{q_type}: {stats['correct']}/{stats['total']} ({type_accuracy:.2f}%)")
            
            print("="*50)
            
            # 输出详细结果
            print("\n详细结果:")
            for result in user_detail_results:  
                try:
                    print(f"用户 {result.get('index', 'unknown')}: {'✓' if result.get('is_correct', False) else '✗'}")
                    print(f"  问题类型: {result.get('question_type', 'unknown')}")
                    print(f"  问题: {result.get('question', 'N/A')}")
                    print(f"  上下文: {result.get('context', 'N/A')}")
                    print(f"  回答: {result.get('answer', 'N/A')}...")
                    print(f"  标准答案: {result.get('golden_answer', 'N/A')}...")
                    print(f"  记忆操作: {result.get('counts', {})}")
                    print()
                except Exception as e:
                    print(f"打印结果时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                    print()
                
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()