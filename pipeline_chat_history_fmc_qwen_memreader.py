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
from vector_db import VectorDBConfig, VectorDBFactory, QdrantDB
# ==========================================
# 0. Setup & Prompts
# ==========================================
load_dotenv()

# ⚠️ 请确保环境变量中有 OPENAI_API_KEY 和 MILVUS_URI
# 如果是本地测试，确保 Docker 中 Milvus 已启动

# Select provider: "openai" or "gemini"
# LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Select LLM mode: "local" or "online"
LLM_MODE = os.getenv("LLM_MODE", "online")

if LLM_MODE == "local":
    # Initialize client for Local LLM
    GENERATION_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-32B")
    llm_client = OpenAI(
        api_key=os.getenv("LOCAL_LLM_API_KEY", "EMPTY"), 
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://0.0.0.0:8088/v1")
    )
    print(f"🚀 Using Local LLM for generation: {llm_client.base_url}, model: {GENERATION_MODEL}")
else:
    # Initialize client for OpenAI (Online)
    GENERATION_MODEL = os.getenv("ONLINE_LLM_MODEL", "gpt-4o-mini")
    llm_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    print(f"🌐 Using OpenAI for generation. model: {GENERATION_MODEL}")

# Separate Model Configuration
MEMREADER_MODEL = os.getenv("MEMREADER_MODEL", GENERATION_MODEL)
MEMORY_MANAGER_MODEL = os.getenv("MEMORY_MANAGER_MODEL", GENERATION_MODEL)
print(f"🔹 MemReader Model: {MEMREADER_MODEL}")
print(f"🔹 Memory Manager Model: {MEMORY_MANAGER_MODEL}")

# Initialize a separate client for embeddings
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1") # 默认本地API地址

embedding_client = OpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL
)

# Use the model specified by user
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024")) # Default to 1024 for Qwen3-0.6B
print(f"🔹 Embedding Model: {EMBEDDING_MODEL}, Dimension: {EMBEDDING_DIMENSION}")


MEMORY_MANAGER_PROMPT = """You are a specialized Memory Manager Agent.
Your role is to maintain the consistency and growth of a memory graph using the provided tools.

[INPUTS]
You will receive:
1. "New Facts": A list of atomic facts extracted from the latest user input.
2. "Existing Memories": A list of retrieved memory items, each with a simplified Integer ID (e.g., "0", "1", "2").
   - These memories include those directly related to the new facts, as well as other related facts connected with these memories.
   - They form a connected graph of information relevant to the new facts.

[MANDATORY OUTPUT FORMAT]
For every new fact you process, you MUST:
1. First generate a detailed thinking process
2. Then call the appropriate tool

[THINKING PROCESS REQUIREMENTS]
Your thinking process MUST include:
- The specific new fact you're analyzing
- Which existing memories are relevant (with their IDs)
- How memories are connected through related facts
- Your comparison and reasoning
- Which operation you've decided to perform and why

[OPERATIONS & GUIDELINES]
Compare New Facts with Existing Memories and perform the following operations using the available tools. 
DO NOT output raw JSON text. You MUST use the provided function tools.

1. **ADD (create_memory)**
   - **Condition**: If a fact contains completely NEW information not present in Existing Memories.
   - **Action**: Call `create_memory` with a concise summary of the facts, not just a simple concatenation.
   - **Important**: Memory content should be a meaningful and concise summary.
  
2. **UPDATE (update_memory)**
   - **Condition**: If a fact adds detail, corrects, or updates a specific Existing Memory.
   - **Constraint**: You MUST use the Integer ID (e.g., "0") provided in the input as the `target_memory_id`.
   - **Logic**: Merge the old content and new fact into a comprehensive statement, not just a simple concatenation.
   - **Example**:
     - Old (ID="0"): "User likes generic pizza."
     - New Fact: "User loves pepperoni pizza."
     - Action: `update_memory(target_memory_id="0", new_content="User loves pepperoni pizza", ...)`

3. **DELETE (delete_memory)**
   - **Condition**: If a fact explicitly contradicts an Existing Memory (and the new fact is trusted), or if the memory is no longer valid.
   - **Constraint**: Use the Integer ID (e.g., "1") as `target_memory_id`.

4. **INFER (infer_memory)**
   - **Condition**: Look for higher-level insights. If combining "Memory A" and "Memory B" reveals a hidden connection or causality.
   - **Action**: Call `infer_memory`.
   - **Example**:
     - Memory A (ID="2"): "User moved to Singapore."
     - Memory B (ID="3"): "User bought a Type G power adapter."
     - Inference: "User is preparing electronics for Singapore power standards."
     - Action: `infer_memory(source_memory_ids=["2", "3"], inference_content="...")`

5. **NOOP (no_operation)**
   - **Condition**: If the fact is redundant (already exactly covered by memory), similar to existing facts associated with the retrieved memories, or trivial.

[STRICT ID RULES]
- When calling `update_memory` or `delete_memory`, **ONLY** use the string integer IDs (e.g., "0", "1", "2") found in the [EXISTING MEMORIES] list.
- **NEVER** invent a UUID or use an ID that is not in the provided list.
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

FACT_MANAGER_PROMPT = """You are a specialized Fact Manager Agent.
Your role is to manage the lifecycle of atomic facts, ensuring new information is captured and related events are consolidated.

[INPUTS]
You will receive:
1. "New Facts": A list of atomic facts extracted from the latest user input. Each fact is formatted as "YYYY-MM-DD: ... (Details: ...)".
2. "Retrieved Facts": A list of existing facts retrieved from the database that are relevant to the new facts.

[OPERATIONS]
Compare New Facts with Retrieved Facts and perform the following operations:

1. **fact_add**
   - **Condition**: If a fact contains completely NEW information not present in Existing facts.
   - **Action**: Call `fact_add`.

2. **fact_trajectorize**
   - **Condition**: If new and old facts have similar content or are related events (e.g., updates on the same topic, sequential events).
   - **Action**: Combine multiple related events into a single "Trajectory" item. This item must explicitly list the chronological progression of events, preserving all original details (dates, specific items, locations, reasons, etc.) from the source facts. Do NOT just summarize; instead, structure it as a timeline log.
   - **Constraint**: You MUST retain the specific "Details" from each original fact in the trajectory.
   - **Example**: 
     - Old Fact: "2025-02-14: User is looking for a birthday gift for mom. (Reason: Mom's interest in Gardening)"
     - New Fact: "2025-02-16: User bought a set of ceramic pots for mom. (Price: $45, Store: HomeDepot)"
     - Trajectory: "2025-02-14: User is looking for a birthday gift for mom. (Reason: Mom's interest in Gardening) -> 2025-02-16: User purchased a set of ceramic pots for mom. (Price: $45, Store: HomeDepot)"
"""

CORE_MEMORY_MANAGER_PROMPT = """You are a Core Memory Manager Agent.
Your role is to build and maintain a comprehensive, long-term profile of the user by managing the "Core Memory" block.

[WHAT TO EXTRACT AND SAVE]
Analyze the provided facts and context to deeply understand the user. Capture and distill information that defines who they are and how to interact with them:
- **Identity & Personal Details**: Name, age, identity, role, and significant dates. (e.g., "Name is John Doe", "Born on 1990-05-15")
- **Personality & Values**: Traits, characteristics, communication style, and core beliefs. (e.g., "Introverted and analytical", "Values sustainability and non-toxic living")
- **Personal Preferences**: Specific likes/dislikes in food, products, brands, and entertainment. (e.g., "Loves Italian cuisine", "Prefers RPG games over shooters", "Favorite brand is Apple")
- **Activity & Service Preferences**: Preferences for dining, travel, hobbies, and other services. (e.g., "Prefers boutique hotels", "Enjoys morning runs", "Always books window seats")
- **Professional & Career**: Job titles, work habits, and professional goals. (e.g., "Senior Data Scientist at Meta", "Aims to lead an AI research team")
- **Relationships**: Family, friends, colleagues, and key social connections. (e.g., "Has a younger brother named Mike", "Close friend with Emma who is a marathon runner")
- **Plans & Intentions**: Upcoming events, trips, and long-term aspirations. (e.g., "Planning a hiking trip to Yosemite next month", "Wants to build a personal knowledge management system")
- **Health & Wellness**: Dietary restrictions, fitness routines, and wellness habits. (e.g., "Follows a gluten-free diet", "Practices yoga every morning")
- **Behaviors & Habits**: Recurring user behaviors and miscellaneous personal details. (e.g., "Usually reads for 30 minutes before bed", "Always drinks coffee while working")
- **Milestones**: Critical life events and significant milestones. (e.g., "Graduated from MIT in 2012", "Started first company in 2020")
- **Contextual Gold**: Any unique information that enhances future personalization. (e.g., "Is a huge fan of vintage mechanical watches", "Used to live in Paris for 5 years and speaks fluent French")

[EXAMPLES OF GOOD CORE MEMORY ENTRIES]
- "Is a software engineer at Google, specializing in machine learning"
- "Loves to play Cyberpunk 2077, prefers RPG games over shooters"
- "Has publications: 1. Paper on NLP transformers 2. Book on AI Ethics"
- "Close friend: Emma (marathon runner), meets weekly for coffee"
- "Working on long-term project: Building a personal knowledge management system"
- "Personality: Introverted, analytical, values deep conversations over small talk"

[INPUTS]
You will receive:
1. "New Facts": Atomic facts extracted from the current conversation.
2. "Old Core Memory": The existing content of the Core Memory block.
3. "Retrieved Memories": Relevant historical context for reference.

[OPERATIONS]
Analyze the inputs and perform one of the following operations:

1. **core_memory_add**
   - **Condition**: Add new, significant information to the Core Memory block.
   - **Example**:
     - New Fact: "User just started a new job as a Data Scientist at Amazon."
     - Action: `core_memory_add(content="Is a Data Scientist at Amazon (Started 2024)")`

2. **core_memory_update**
   - **Condition**: Update specific outdated or incorrect information. 
   - **Requirement**: You MUST specify both `old_text` and `new_text` for matching.
   - **Example**:
     - Old Text in Core Memory: "Is a software engineer at Google"
     - New Fact: "User has been promoted to Senior Software Engineer at Google."
     - Action: `core_memory_update(old_text="Is a software engineer at Google", new_text="Is a Senior software engineer at Google")`

3. **core_memory_rewrite**
   - **Condition**: Reorganize and consolidate the entire block. Used when its length exceeds 5000 chars or when major updates are needed.
   - **Action**: Completely rewrite the block to be more concise and better organized while preserving all core information.

[FULL PROFILE EXAMPLE]
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
8. Store Assistant's Key Information: Remember important explanations, definitions, project details, or summaries provided by the assistant.


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
# assistant: That sounds like a great challenge. I'll make a note that you are hiking Half Dome with Mike.
# Output: {{"facts" : [
#     {{"fact": "Going to Yosemite National Park with brother Mike", "details": ["Activity: Hiking", "Time: 2025-05-24 to 2025-05-25 (Next weekend)"]}},
#     {{"fact": "Hopes to summit Half Dome", "details": ["Location: Yosemite National Park", "Intent: Hiking goal"]}}
# ]}}

# Input:
# **Today's Date**: 2025-08-10
# **Previous Chat History**:
# assistant: I recall you mentioned being vegetarian. Is that still the case?

# **Current Conversation Turn**:
# user: Actually, no. I've started eating chicken recently for the protein, but I still avoid red meat.
# assistant: Understood. I've updated your dietary preferences to include chicken but exclude red meat.
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
# assistant: That is impressive dedication! Writing 2000 words on a public computer is no small feat.
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
# assistant: Got it. Since she likes rock climbing, I recommend checking out the new indoor climbing gym downtown.
# Output: {{"facts" : [
#     {{"fact": "Wife is into rock climbing", "details": ["Interest: Outdoor sports"]}},
#     {{"fact": "Wife already has tennis gear", "details": []}},
#     {{"fact": "Recommends new indoor climbing gym downtown", "details": ["Reason: Matches wife's rock climbing interest"]}}
# ]}}

# Input:
# **Today's Date**: 2026-02-01
# **Previous Chat History**:
# user: I need to book a flight to London.
# assistant: When are you planning to fly?

# **Current Conversation Turn**:
# user: I want to leave on the 15th and return on the 22nd. I strictly want to avoid overnight layovers.
# assistant: Noted. I will filter for direct flights. Just a reminder: London is currently 8 hours ahead of your timezone.
# Output: {{"facts" : [
#     {{"fact": "Planning a flight to London", "details": ["Departure: 2026-02-15", "Return: 2026-02-22"]}},
#     {{"fact": "Strictly avoids overnight layovers", "details": ["Preference: Flight booking constraint"]}},
#     {{"fact": "London is 8 hours ahead of user's timezone", "details": ["Context: Timezone reminder"]}}
# ]}}

# Return the facts and preferences in a json format as shown above.

# Remember the following:
# - Today's date is {today_date}.
# - **Supplementary Details**: The `details` list must act as **METADATA** to supplement the fact (e.g., Time, Location, Price, Platform, Reason), **NOT** just splitting the fact's words.
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

# --- TOOLS ---
MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_memory",
            "description": "Create a NEW independent memory node with a concise summary of the facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The concise summary content of the new memory, not just a list of facts."}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Update an existing memory by merging the old content and new fact into a comprehensive, concise statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_memory_id": {"type": "string", "description": "The simplified Integer ID (e.g., '0') of the memory to update, found in the [EXISTING MEMORIES] list."},
                    "new_content": {"type": "string", "description": "The merged/updated comprehensive statement."}
                },
                "required": ["target_memory_id", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "infer_memory",
            "description": "Look for higher-level insights. If combining multiple existing memories reveals a hidden connection or causality, create an inferred memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_memory_ids": {"type": "array", "items": {"type": "string"}, "description": "List of simplified Integer IDs (e.g., ['0', '1']) acting as premises, found in the [EXISTING MEMORIES] list."},
                    "inference_content": {"type": "string", "description": "The higher-level insight or inference derived from combining the source memories."}
                },
                "required": ["source_memory_ids", "inference_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Archive/Soft-delete a memory if it explicitly contradicts a new fact (and the new fact is trusted), or if the memory is no longer valid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_memory_id": {"type": "string", "description": "The simplified Integer ID (e.g., '1') of the memory to delete, found in the [EXISTING MEMORIES] list."}
                },
                "required": ["target_memory_id"]
            }
        }
    },
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

FACT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fact_add",
            "description": "Register a completely new fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content of the new fact."},
                    "details": {"type": "array", "items": {"type": "string"}, "description": "Details or metadata of the fact."}
                },
                "required": ["content", "details"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fact_trajectorize",
            "description": "Create timeline with timestamp range, synthesizing events and drawing well supported conclusions from patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "related_fact_ids": {"type": "array", "items": {"type": "string"}, "description": "IDs of the old facts to be archived."},
                    "content": {"type": "string", "description": "The combined timeline summary including timestamp range."}
                },
                "required": ["related_fact_ids", "content"]
            }
        }
    },
]

CORE_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "core_memory_add",
            "description": "Add new information to the Core Memory block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The new information to add."}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_update",
            "description": "Update specific information in the Core Memory block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_text": {"type": "string", "description": "The specific text to be replaced."},
                    "new_text": {"type": "string", "description": "The new text to replace with."}
                },
                "required": ["old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_rewrite",
            "description": "Rewrite the entire Core Memory block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_block_content": {"type": "string", "description": "The completely new content for the Core Memory block."}
                },
                "required": ["new_block_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "no_operation",
            "description": "No action needed if the core memory is up to date.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "The reason for no operation."}},
                "required": ["reason"]
            }
        }
    }
]

# --- UTILS ---
def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    # Use embedding_client (always OpenAI) instead of llm_client (which might be local)
    try:
        return embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSION).data[0].embedding
    except Exception:
        # Fallback if dimensions arg is not supported or other error, try without dimensions
        return embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

@dataclass
class MilvusConfig:
    """Milvus配置类（兼容旧代码）"""
    uri: str = os.getenv("MILVUS_URI")
    user_name: str = os.getenv("MILVUS_USER_NAME")
    # password: str = os.getenv("MILVUS_PASSWORD")
    db_name: str = os.getenv("MILVUS_DB_NAME", "default")
    dimension: int = EMBEDDING_DIMENSION
    
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
class LocalJsonlDB:
    def __init__(self, storage_dir="./local_mem_db"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.collections = {}
        # DataType wrapper to mimic Milvus/VectorDB interface
        class DataType:
            VARCHAR = "VARCHAR"
            FLOAT_VECTOR = "FLOAT_VECTOR"
            INT64 = "INT64"
            JSON = "JSON"
        self.DataType = DataType()

    def create_schema(self, auto_id=False, enable_dynamic_field=True):
        class Schema:
            def add_field(self, *args, **kwargs): pass
        return Schema()

    def prepare_index_params(self):
        class IndexParams:
            def add_index(self, **kwargs): pass
        return IndexParams()

    def has_collection(self, name):
        return os.path.exists(os.path.join(self.storage_dir, f"{name}.jsonl"))

    def create_collection(self, name, schema=None):
        path = os.path.join(self.storage_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                pass
        self.collections[name] = []
        print(f"[LocalJsonlDB] Collection {name} created at {path}")

    def create_index(self, collection_name, index_params=None):
        pass # No-op for JSONL

    def drop_collection(self, name):
        path = os.path.join(self.storage_dir, f"{name}.jsonl")
        if os.path.exists(path):
            os.remove(path)
        if name in self.collections:
            del self.collections[name]
        print(f"[LocalJsonlDB] Dropped collection {name}")

    def load_collection(self, name):
        path = os.path.join(self.storage_dir, f"{name}.jsonl")
        data = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except:
                            continue
        self.collections[name] = data
        print(f"[LocalJsonlDB] Loaded {len(data)} items from {name}")

    def upsert(self, collection_name, data: List[Dict]):
        if collection_name not in self.collections:
            self.load_collection(collection_name)
        
        path = os.path.join(self.storage_dir, f"{collection_name}.jsonl")
        
        # Simple upsert: remove existing items with same ID and append new ones
        # Identify primary key based on collection name convention or assume specific keys
        # Fact: fact_id, Memory: memory_id, Chunk: chunk_id
        pk_field = "id"
        if "fact" in collection_name: pk_field = "fact_id"
        elif "memories" in collection_name: pk_field = "memory_id"
        elif "chunk" in collection_name: pk_field = "chunk_id"
        
        new_ids = {item.get(pk_field) for item in data if item.get(pk_field)}
        
        # Filter out existing items that are being updated
        existing = self.collections[collection_name]
        kept_items = [item for item in existing if item.get(pk_field) not in new_ids]
        
        # Add new items
        updated_list = kept_items + data
        self.collections[collection_name] = updated_list
        
        # Rewrite file (inefficient for large data but functional for local test)
        with open(path, 'w', encoding='utf-8') as f:
            for item in updated_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"[LocalJsonlDB] Upserted {len(data)} items into {collection_name}")

    def _match_filter(self, item, filter_str):
        """Helper to evaluate filter string against an item"""
        if not filter_str:
            return True
        
        # Split by ' and ' to handle multiple conditions
        # Note: This is a simple parser and assumes ' and ' is the only logical operator used
        conditions = filter_str.split(' and ')
        
        for cond in conditions:
            cond = cond.strip()
            if not cond:
                continue
                
            if " in [" in cond:
                # Handle "field in ['a', 'b']"
                try:
                    parts = cond.split(" in [")
                    field = parts[0].strip()
                    val_content = parts[1].split("]")[0]
                    # Handle quoted strings in list
                    vals = [v.strip().strip("'").strip('"') for v in val_content.split(",")]
                    if str(item.get(field)) not in vals:
                        return False
                except:
                    print(f"⚠️ Filter parse error (IN): {cond}")
                    return False
                    
            elif "==" in cond:
                # Handle "field == 'value'"
                try:
                    parts = cond.split("==")
                    field = parts[0].strip()
                    val = parts[1].strip().strip("'").strip('"')
                    if str(item.get(field)) != val:
                        return False
                except:
                    print(f"⚠️ Filter parse error (==): {cond}")
                    return False
            else:
                # Ignore unsupported or complex filters for now, defaulting to match
                # (This might include non-filtering data, but safer than crashing)
                pass
                
        return True

    def search(self, collection_name, vectors, filter=None, limit=5, output_fields=None, similarity_threshold=None):
        if collection_name not in self.collections:
            self.load_collection(collection_name)
            
        items = self.collections[collection_name]
        if not items:
            return [[]]
            
        # Filter items using improved matcher
        filtered_items = [item for item in items if self._match_filter(item, filter)]
                
        if not filtered_items:
            return [[]]

        # Calculate cosine similarity
        query_vec = np.array(vectors[0])
        results = []
        
        for item in filtered_items:
            item_vec = item.get("embedding")
            if not item_vec:
                continue
            item_vec = np.array(item_vec)
            
            # Cosine similarity
            norm_q = np.linalg.norm(query_vec)
            norm_i = np.linalg.norm(item_vec)
            if norm_q == 0 or norm_i == 0:
                score = 0
            else:
                score = np.dot(query_vec, item_vec) / (norm_q * norm_i)
            
            if similarity_threshold and score < similarity_threshold:
                continue
                
            # Create result object
            res = {"entity": {k: item.get(k) for k in (output_fields or item.keys())}, "distance": float(score), "id": item.get("id")}
            # Also keep all fields in entity for convenience if output_fields is missing key stuff
            if output_fields:
                 for k in output_fields:
                     res['entity'][k] = item.get(k)
            else:
                 res['entity'] = item
            
            results.append(res)
            
        # Sort by score descending
        results.sort(key=lambda x: x['distance'], reverse=True)
        return [results[:limit]]

    def query(self, collection_name, filter=None, output_fields=None, limit=None):
        if collection_name not in self.collections:
            self.load_collection(collection_name)
            
        items = self.collections[collection_name]
        
        # Filter items using improved matcher
        filtered_items = [item for item in items if self._match_filter(item, filter)]
        
        results = []
        for item in filtered_items:
            if output_fields:
                results.append({k: item.get(k) for k in output_fields})
            else:
                results.append(item)
                
        return results

    def delete(self, collection_name, filter=None):
        """Simulate delete by filtering out items"""
        if collection_name not in self.collections:
            return
            
        items = self.collections[collection_name]
        # Keep items that DO NOT match the filter
        kept_items = [item for item in items if not self._match_filter(item, filter)]
        
        self.collections[collection_name] = kept_items
        
        # Rewrite file
        path = os.path.join(self.storage_dir, f"{collection_name}.jsonl")
        with open(path, 'w', encoding='utf-8') as f:
            for item in kept_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[LocalJsonlDB] Deleted items from {collection_name}, remaining: {len(kept_items)}")

class MemoryPipeline:
    def __init__(self, config=None, vector_db_type="milvus", clear_db=False, mode='eval', dataset_name="", extract_only=False):
        """初始化MemoryPipeline
        
        Args:
            config: MilvusConfig或VectorDBConfig实例，如果为None则使用默认配置
            vector_db_type: 指定使用的向量数据库类型，支持"milvus"或"qdrant"
            clear_db: 是否清空数据库，默认为False
            dataset_name: 数据集名称，用于集合名称后缀，默认为空
            extract_only: 是否仅进行提取，跳过后续的预处理、检索和执行步骤
        """
        self.extract_only = extract_only

        # 如果没有提供配置，创建默认配置
        if config is None:
            config = VectorDBConfig(uri="local")
        
        self.config = config
        
        # 使用本地 JSONL DB 替代 VectorDBFactory
        print("🚀 Using Local JSONL Database storage_dir='./local_mem_db'")
        self.client = LocalJsonlDB(storage_dir="./local_mem_db")
        
        # 根据模式和数据集名称设置集合名称

        base_suffix = "_test" if mode == 'test' else ""
        dataset_suffix = f"_{dataset_name}" if dataset_name else ""
        full_suffix = f"{base_suffix}{dataset_suffix}"
        
        self.semantic_col = f"memories{full_suffix}_v1"
        self.fact_col = f"facts{full_suffix}_v1"
        self.chunk_col = f"chunks{full_suffix}_v1"
        
        self.dim = self.config.dimension  # Save dimension as instance variable
        # 初始化操作次数计数器
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        # 添加带历史支持的memreader prompt
        self.MEMREADER_PROMPT_WITH_HISTORY = MEMREADER_PROMPT_WITH_HISTORY
        
        # Initialize Core Memory
        self.core_memory = ""
        # 初始化 Core Memory 存储路径
        if hasattr(self.client, 'storage_dir'):
            self.core_memory_file = os.path.join(self.client.storage_dir, "core_memory.json")
        else:
            self.core_memory_file = "core_memory.json"
        
        self._init_collections(clear_db=clear_db)
         
        # 如果需要清空数据库，同时清空 Core Memory 文件
        if clear_db and os.path.exists(self.core_memory_file):
            try:
                os.remove(self.core_memory_file)
                print(f"   🧹 Cleared Core Memory file: {self.core_memory_file}")
            except Exception as e:
                print(f"   ⚠️ Failed to clear Core Memory file: {e}")

    def _load_core_memory(self, user_id: str):
        """加载特定用户的 Core Memory"""
        self.core_memory = ""
        if os.path.exists(self.core_memory_file):
            try:
                with open(self.core_memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.core_memory = data.get(user_id, "")
                print(f"   🧠 Loaded Core Memory for {user_id} ({len(self.core_memory)} chars)")
            except Exception as e:
                print(f"   ⚠️ Error loading Core Memory: {e}")

    def _save_core_memory(self, user_id: str):
        """保存特定用户的 Core Memory"""
        data = {}
        if os.path.exists(self.core_memory_file):
            try:
                with open(self.core_memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        
        data[user_id] = self.core_memory
        
        try:
            with open(self.core_memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"   💾 Saved Core Memory for {user_id}")
        except Exception as e:
            print(f"   ⚠️ Error saving Core Memory: {e}")

    def _format_context(self, core_memory, retrieved_memories):
        """
        统一格式化上下文，包含 Core Memory, Semantic Memory 和 Facts。
        """
        context_parts = []
        context_parts.append("### CORE MEMORY ###")
        # 1. Core Memory
        if core_memory and core_memory.strip():
            # context_parts.append("### CORE MEMORY ###")
            context_parts.append(core_memory.strip())
            context_parts.append("") # 换行
            
        # 分离 Semantic Memory 和 Facts
        memories = [item for item in retrieved_memories if item.get("type", "memory") == "memory"]
        facts = [item for item in retrieved_memories if item.get("type") == "fact"]
        
        # 2. Semantic Memory
        context_parts.append("### SEMANTIC MEMORY ###")
        if memories:
            # context_parts.append("### SEMANTIC MEMORY ###")
            for mem in memories:
                # 简化时间戳格式: YYYY-MM-DD HH:MM
                try:
                    ts_str = datetime.fromtimestamp(mem['created_at'], timezone.utc).strftime('%Y-%m-%d %H:%M')
                except:
                    ts_str = "Unknown Time"
                context_parts.append(f"- {ts_str}: {mem['content']}")
            context_parts.append("") # 换行
            
        # 3. Facts
        context_parts.append("### FACTS ###")
        if facts:
            # context_parts.append("### FACTS ###")
            for fact in facts:
                # 简化时间戳格式: YYYY-MM-DD HH:MM
                try:
                    ts_str = datetime.fromtimestamp(fact['created_at'], timezone.utc).strftime('%Y-%m-%d %H:%M')
                except:
                    ts_str = "Unknown Time"
                
                # 格式化细节为 (Detail: xxx) 形式并内联
                details = fact.get("details", [])
                if details:
                    if isinstance(details, list):
                        details_str = "; ".join(details)
                    else:
                        details_str = str(details)
                    
                    if len(details_str) > 150:
                        details_str = details_str[:150] + "..."
                    
                    # 内联格式: 时间: 内容 (Detail: 细节)
                    context_parts.append(f"- {ts_str}: {fact['content']} (Detail: {details_str})")
                else:
                    context_parts.append(f"- {ts_str}: {fact['content']}")
            context_parts.append("") # 换行
            
        return "\n".join(context_parts).strip()


    def _init_collections(self, clear_db=False):
        dim = self.config.dimension
        
        # 如果需要清空数据库，先删除所有集合
        if clear_db:
            print("正在清空数据库...")
            # 直接删除集合，不检查存在性
            self.client.drop_collection(self.semantic_col)
            self.client.drop_collection(self.fact_col)
            self.client.drop_collection(self.chunk_col)
            print("数据库清空完成.")
        
        # 检查并创建集合
        
        # 处理 memories 集合
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
                    idx_params = self.client.prepare_index_params()
                    idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                    self.client.create_index(self.semantic_col, index_params=idx_params)
                    print(f"集合 '{self.semantic_col}' 的索引创建成功或已存在")
                except Exception as e:
                    print(f"创建索引失败: {e}")
            else:
                print(f"Collection '{self.semantic_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.semantic_col)
            print(f"Collection '{self.semantic_col}' created or exists.")
        
        # 处理 facts 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.fact_col):
                s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                s.add_field("fact_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
                s.add_field("linked_chunk_id", self.client.DataType.VARCHAR, max_length=64)
                s.add_field("text", self.client.DataType.VARCHAR, max_length=65535)
                s.add_field("details", self.client.DataType.JSON)  # 添加details字段
                s.add_field("timestamp", self.client.DataType.INT64)
                s.add_field("user_id", self.client.DataType.VARCHAR, max_length=64)  # 添加user_id字段
                s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
                
                # 创建集合
                self.client.create_collection(self.fact_col, schema=s)
                print(f"Collection '{self.fact_col}' created.")
                
                # 直接创建索引，不检查索引是否存在
                # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
                try:
                    print(f"为集合 '{self.fact_col}' 创建索引...")
                    idx_params = self.client.prepare_index_params()
                    idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                    self.client.create_index(self.fact_col, index_params=idx_params)
                    print(f"集合 '{self.fact_col}' 的索引创建成功或已存在")
                except Exception as e:
                    print(f"创建索引失败: {e}")
            else:
                print(f"Collection '{self.fact_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.fact_col)
            print(f"Collection '{self.fact_col}' created or exists.")
        
        # 处理 chunks 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.chunk_col):
                s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                s.add_field("chunk_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
                s.add_field("text", self.client.DataType.VARCHAR, max_length=65535)
                s.add_field("timestamp", self.client.DataType.INT64)
                s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
                
                # 创建集合
                self.client.create_collection(self.chunk_col, schema=s)
                print(f"Collection '{self.chunk_col}' created.")
                
                # 直接创建索引，不检查索引是否存在
                # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
                try:
                    print(f"为集合 '{self.chunk_col}' 创建索引...")
                    idx_params = self.client.prepare_index_params()
                    idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                    self.client.create_index(self.chunk_col, index_params=idx_params)
                    print(f"集合 '{self.chunk_col}' 的索引创建成功或已存在")
                except Exception as e:
                    print(f"创建索引失败: {e}")
            else:
                print(f"Collection '{self.chunk_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.chunk_col)
            print(f"Collection '{self.chunk_col}' created or exists.")
        
        # 直接加载所有集合，不进行复杂的错误处理
        print("Loading collections into memory...")
        
        # 加载集合（Qdrant 不需要显式加载）
        if hasattr(self.client, 'load_collection'):
            # 为每个集合创建索引后直接加载
            print(f"加载集合 '{self.semantic_col}'...")
            self.client.load_collection(self.semantic_col)
            
            print(f"加载集合 '{self.fact_col}'...")
            self.client.load_collection(self.fact_col)
            
            print(f"加载集合 '{self.chunk_col}'...")
            self.client.load_collection(self.chunk_col)
            
            print("All collections loaded successfully.")

    # --- Step 1: Extract ---
    def _log_extraction(self, chunk_text: str, facts: List[Dict], turn_details: List[Dict] = None):
        """记录提取的事实到日志文件"""
        log_file = "memreader_log.jsonl"
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_text": chunk_text,
                "extracted_facts": facts
            }
            if turn_details:
                entry["turn_details"] = turn_details
                
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"   📝 [User Request] Logged extracted facts to {log_file}")
        except Exception as e:
            print(f"   ⚠️ Logging failed: {e}")

    def step_extract(self, session_or_text, extract_mode: str = "whole", timestamp: int = None, max_history_turns: int = 5, dataset_question: str = "") -> Dict:
        """
        从对话中提取事实
        
        Args:
            session_or_text: 对话会话，可以是原始session list或对话文本
            extract_mode: 提取模式，可选值：
                - "whole": 对整个chunk进行提取
                - "turn": 按轮次提取，每轮user-assistant对话单独提取，并附上chat history
            timestamp: 时间戳，可选，默认使用当前时间
            max_history_turns: 聊天历史的最大轮数，仅在extract_mode="turn"时生效
            dataset_question: 数据集中的问题，用于日志记录
        
        Returns:
            包含提取事实的字典
        """
        # print(f"\n👀 [1. Extract] Processing...")
        
        # 如果没有提供timestamp，使用当前时间
        if timestamp is None:
            timestamp = int(time.time())
        
        # 如果是按轮次提取，直接处理session list
        if extract_mode == "turn" and isinstance(session_or_text, list):
            try:
                all_facts = []
                chat_history = []  # 保存完整的对话历史
                session_turn_details = [] # 保存每一轮的详细提取结果
                
                # 遍历session list，动态识别turn
                i = 0
                while i < len(session_or_text):
                    msg = session_or_text[i]
                    turn = []
                    # 无论谁说话，都将其视为一个待提取的Turn（或者尝试合并）
                    # 策略：以任何消息开始，如果下一条是对方的回复，则合并为一个 Turn
                    
                    turn.append(msg)
                    current_role = msg.get("role")
                    
                    # 尝试合并下一条消息（如果是对话流的话）
                    # 规则：User -> Assistant 或 Assistant -> User 可以合并
                    # 但如果是 System 消息，通常单独处理
                    if current_role != "system" and i + 1 < len(session_or_text):
                        next_msg = session_or_text[i+1]
                        next_role = next_msg.get("role")
                        
                        # 如果角色不同且下一条也不是 System，则合并
                        if next_role != "system" and next_role != current_role:
                            turn.append(next_msg)
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                    
                    # 添加当前turn到chat history
                    chat_history.append(turn)
                    
                    # 对所有 Turn 都进行事实提取（不仅仅是 User Turn）
                    # 将turn转换为文本格式
                    turn_text = parse_messages(turn)
                    
                    # 构建聊天历史，使用当前turn之前的max_history_turns轮对话
                    history_turns = chat_history[:-1][-max_history_turns:]  # 最近max_history_turns轮历史
                    history_text = parse_messages([m for t in history_turns for m in t])
                    
                    # 对单轮对话提取事实，传递timestamp和chat_history参数
                    turn_facts = self._extract_single_turn(turn_text, timestamp, history_text)
                    
                    # 记录该轮的详细信息
                    session_turn_details.append({
                        "turn_idx": len(chat_history),
                        "question": dataset_question,
                        "turn_text": turn_text,
                        "facts": turn_facts
                    })
                    
                    # 为每个事实添加轮次信息和chat history引用
                    for fact in turn_facts:
                            fact["turn_idx"] = len(chat_history)  # 轮次从1开始
                            fact["has_history"] = len(history_text) > 0
                            fact["history_turns"] = len(history_turns)  # 聊天历史的轮数
                            all_facts.extend(turn_facts)
                    
                # 将session转换为文本格式，用于返回
                chunk_text = parse_messages(session_or_text)
                self._log_extraction(chunk_text, all_facts, turn_details=session_turn_details)
                return {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text, "new_facts": all_facts, "timestamp": timestamp, "chat_history": chat_history}
            except Exception as e:
                print(f"按轮次处理session失败，回退到whole模式: {e}")
        
        # 默认模式：对整个session或文本进行提取，传递timestamp参数
        if isinstance(session_or_text, list):
            chunk_text = parse_messages(session_or_text)
        else:
            chunk_text = session_or_text
            
        facts = self._extract_single_turn(chunk_text, timestamp)
        self._log_extraction(chunk_text, facts)
        return {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text, "new_facts": facts, "timestamp": timestamp, "chat_history": [chunk_text]}
    
    def _extract_single_turn(self, text: str, timestamp: int = None, chat_history: str = "") -> List[Dict]:
        """
        对单个文本片段提取事实
        
        Args:
            text: 要提取事实的文本
            timestamp: 时间戳，可选，默认使用当前时间
            chat_history: 之前的对话历史，用于提供上下文
            
        Returns:
            提取到的事实列表
        """
        try:
            # 将timestamp转换为YYYY-MM-DD格式的日期字符串
            if timestamp is None:
                today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            else:
                today_date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")
            
            # 替换prompt中的today_date占位符
            # 优先使用带历史的prompt，如果不存在则使用原prompt
            prompt = getattr(self, 'MEMREADER_PROMPT_WITH_HISTORY', MEMREADER_PROMPT)
            formatted_prompt = prompt.format(today_date=today_date)
            
            # 构建用户输入，包含chat history和当前对话
            user_input = ""
            if chat_history:
                user_input += f"Previous Chat History:\n{chat_history}\n\n"
            user_input += f"Current Conversation Turn:\n{text}"
            
            # 检查 user_input 是否为空，避免 API 报错
            if not user_input.strip():
                print("⚠️ Warning: user_input is empty, skipping extraction.")
                return []
            
            # 再次检查 formatted_prompt 是否为空
            if not formatted_prompt.strip():
                print("⚠️ Warning: formatted_prompt is empty, skipping extraction.")
                return []
            
            max_retries = 3
            fact_objects = []
            
            import traceback # Import here to ensure availability
            
            for attempt in range(max_retries):
                try:
                    # 尝试调用 LLM
                    # 注意：部分本地模型可能不支持 response_format={"type": "json_object"}
                    # 这里做一个简单的兼容性处理
                    try:
                        response = llm_client.chat.completions.create(
                            model=MEMREADER_MODEL,
                            messages=[
                                    {"role": "system", "content": formatted_prompt}, 
                                    {"role": "user", "content": user_input}],
                            response_format={"type": "json_object"}, 
                            extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                            },
                            temperature=0
                        )
                    except Exception as e_format:
                        # 如果是参数错误（如不支持 json_object），尝试不带 format 参数
                        # print(f"⚠️ Attempt {attempt + 1}: 'json_object' format might not be supported, retrying without it. Error: {e_format}")
                        response = llm_client.chat.completions.create(
                            model=MEMREADER_MODEL,
                            messages=[
                                    {"role": "system", "content": formatted_prompt}, 
                                    {"role": "user", "content": user_input}],
                            extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                            },
                            temperature=0
                        )

                    raw_content = response.choices[0].message.content
                    
                    if not raw_content:
                        print(f"⚠️ Attempt {attempt + 1}/{max_retries}: Received empty response from LLM.")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            raise ValueError("Received empty response from LLM after max retries")

                    json_str = extract_json(raw_content)
                    fact_objects = json.loads(json_str).get("facts", [])
                    break  # Success, exit loop
                    
                except json.JSONDecodeError:
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries}: JSON Decode Error. Raw content: {raw_content}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries}: API Error: {e}")
                    traceback.print_exc()
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise
            
            # 保留完整的fact对象，包括details信息
            facts = []
            for fact_obj in fact_objects:
                if fact_obj.get("fact"):
                    facts.append({
                        "text": fact_obj.get("fact", ""),
                        "details": fact_obj.get("details", []),
                        "timestamp": timestamp,  # 添加时间戳
                        "chat_history_length": len(chat_history.split("\n")) if chat_history else 0  # 添加历史长度
                    })
        except Exception as e: 
            print(f"❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            # 失败时返回空列表，而不是原始文本，避免污染日志
            facts = []
        return facts

    # --- Step 2: Retrieve ---    
    def step_retrieve(self, extract_result: Dict, limit: int = 3, user_id: str = 'default', similarity_threshold: float = None) -> List[Dict]:
        new_facts = extract_result['new_facts']
        if not new_facts: return []
        
        print(f"🔍 [2. Retrieve] Searching Memories & Facts for {len(new_facts)} facts...")
        context_bundles = []

        for fact in new_facts:
            query_vec = get_embedding(fact['text'])
            
            # 1. 检索相关记忆 (Candidates)
            res_mem = self.client.search(
                self.semantic_col, [query_vec], filter=f"status == 'active' and user_id == '{user_id}'", limit=limit,
                output_fields=["content", "memory_id", "created_at"],
                similarity_threshold=similarity_threshold
            )
            candidates = []
            if res_mem and res_mem[0]:
                for hit in res_mem[0]:
                    candidates.append(hit['entity'])
            
            # 2. 🌟 直接从 fact_col 检索相关事实 (Related Facts)
            # 不再依赖 memory-fact 的关联，改为语义检索事实
            res_fact = self.client.search(
                self.fact_col, [query_vec], filter=f"user_id == '{user_id}'", limit=limit,
                output_fields=["fact_id", "text", "timestamp", "details"],
                similarity_threshold=similarity_threshold
            )
            related_facts = []
            if res_fact and res_fact[0]:
                for hit in res_fact[0]:
                    # 检查是否不是 Status: Archived
                    entity = hit['entity']
                    details = entity.get('details', [])
                    if "Status: Archived" not in details:
                        related_facts.append(entity)
            
            context_bundles.append({
                "new_fact": fact,
                "candidates": candidates,
                "related_facts": related_facts  # 这里的 related_facts 是直接检索出来的
            })
            
        return context_bundles

    # --- Step 3: Decide (With ID Mapping) ---
    def step_decide(self, extract_result: Dict, context_bundles: List[Dict], user_id: str = 'default', training_mode: bool = False) -> List[Dict]:
        all_new_facts = extract_result['new_facts']
        
        # 1. 合并去重 Candidates & Related Facts
        temp_mem_storage = {}
        related_facts_storage = {}
        
        for bundle in context_bundles:
            # 合并记忆候选
            for mem in bundle['candidates']:
                temp_mem_storage[mem['memory_id']] = mem
            
            # 合并事实候选 (直接从 step_retrieve 检索出来的)
            for fact in bundle.get('related_facts', []):
                related_facts_storage[fact['fact_id']] = fact
        
        unique_memories_list = list(temp_mem_storage.values())
        unique_related_facts = list(related_facts_storage.values())
        
        if not training_mode:
            print(f"🧠 [3. Manager] Global Decide: {len(all_new_facts)} new facts, {len(unique_memories_list)} memories, {len(unique_related_facts)} related facts.")

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
                candidates_str += f"[Memory Item ID: {simple_id}]\n- Content: {mem['content']}\n\n"
                # 🌟 注意：这里不再展示 Related Facts，因为它们不再关联

        # 构造 Fact Manager 的 Retrieved Facts 字符串
        retrieved_facts_str = ""
        fact_uuid_mapping = {}
        for idx, fact in enumerate(unique_related_facts):
            simple_id = str(idx)
            fact_uuid_mapping[simple_id] = fact['fact_id']
            
            # 格式化时间戳
            ts = fact.get('timestamp')
            date_str = "Unknown Date"
            if ts:
                date_str = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
            
            fact_str = f"{date_str}: {fact['text']}"
            if fact.get('details'):
                details_content = fact['details']
                if isinstance(details_content, list):
                    details_str = "; ".join([str(d) for d in details_content])
                else:
                    details_str = str(details_content)
                fact_str += f" ({details_str})"
            
            retrieved_facts_str += f"[Fact ID: {simple_id}]\n- Content: {fact_str}\n\n"

        # 3. 准备 Prompt 输入
        # 构建包含timestamp和details的facts列表
        formatted_facts = []
        for fact in all_new_facts:
            # 格式化时间戳
            ts = fact.get('timestamp')
            date_str = "Unknown Date"
            if ts:
                date_str = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
            
            fact_str = f"{date_str}: {fact['text']}"
            
            if fact.get('details'):
                details_content = fact['details']
                if isinstance(details_content, list):
                    details_str = "; ".join([str(d) for d in details_content])
                else:
                    details_str = str(details_content)
                fact_str += f" ({details_str})"
            formatted_facts.append(fact_str)
            
        new_facts_json = json.dumps(formatted_facts, ensure_ascii=False)
        
        # A. Fact Manager Input
        fact_user_content = f"""
        [New Facts Stream]
        {new_facts_json}
        
        [Retrieved Facts]
        {retrieved_facts_str}
        """
        
        # B. Memory Manager Input (Existing)
        memory_user_content = f"""
        [New Facts Stream]
        {new_facts_json}
        
        [EXISTING MEMORIES]
        {candidates_str}
        """
        
        # C. Core Memory Manager Input
        core_memory_user_content = f"""
        [New Facts Stream]
        {new_facts_json}
        
        [Retrieved Memories]
        {candidates_str}

        [Current Core Memory Stats]
        - Length: {len(self.core_memory)} characters
        - Limit: 5000 characters (Rewrite recommended if exceeded)

        [Old Core Memory]
        {self.core_memory}
        """

        # 4. 定义并行调用的函数
        def call_agent(system_prompt, user_content, tools, tool_choice="required"):
            try:
                response = llm_client.chat.completions.create(
                    # model="gpt-4o",
                    model=MEMORY_MANAGER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=0
                )
                if not response.choices or not response.choices[0].message.tool_calls:
                    return []
                return response.choices[0].message.tool_calls
            except Exception as e:
                if not training_mode:
                    print(f"   ⚠️ Agent Call Error: {e}")
                return []

        # 5. 并行执行
        all_decisions = []
        
        if not training_mode:
            print("   🚀 Launching 3 parallel Memory Agents...")
            
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_fact = executor.submit(call_agent, FACT_MANAGER_PROMPT, fact_user_content, FACT_TOOLS)
            future_mem = executor.submit(call_agent, MEMORY_MANAGER_PROMPT, memory_user_content, MEMORY_TOOLS)
            future_core = executor.submit(call_agent, CORE_MEMORY_MANAGER_PROMPT, core_memory_user_content, CORE_MEMORY_TOOLS)
            
            fact_calls = future_fact.result()
            mem_calls = future_mem.result()
            core_calls = future_core.result()

        # 6. 解析结果
        
        # 解析 Fact Manager 结果
        for tool_call in fact_calls:
            try:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                decision = {"action": "NOOP", "user_id": user_id, "source": "fact_manager"}
                
                if func_name == "fact_add":
                    decision.update({"action": "FACT_ADD", "summary": args.get("content", ""), "details": args.get("details", [])})
                elif func_name == "fact_trajectorize":
                     related_simple_ids = args.get("related_fact_ids", [])
                     real_related_ids = [fact_uuid_mapping.get(sid) for sid in related_simple_ids if fact_uuid_mapping.get(sid)]
                     
                     decision.update({
                         "action": "FACT_TRAJECTORIZE",
                         "related_fact_ids": real_related_ids,
                         "content": args.get("content", "")
                     })
                
                if decision["action"] != "NOOP":
                    all_decisions.append(decision)
            except Exception as e:
                if not training_mode: print(f"Error parsing fact tool: {e}")

        # 解析 Core Memory Manager 结果
        for tool_call in core_calls:
            try:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                decision = {"action": "NOOP", "user_id": user_id, "source": "core_memory_manager"}
                
                if func_name == "core_memory_add":
                    decision.update({"action": "CORE_MEMORY_ADD", "content": args.get("content", "")})
                elif func_name == "core_memory_update":
                    decision.update({"action": "CORE_MEMORY_UPDATE", "old_text": args.get("old_text", ""), "new_text": args.get("new_text", "")})
                elif func_name == "core_memory_rewrite":
                    decision.update({"action": "CORE_MEMORY_REWRITE", "new_block_content": args.get("new_block_content", "")})
                
                if decision["action"] != "NOOP":
                    all_decisions.append(decision)
            except Exception as e:
                if not training_mode: print(f"Error parsing core memory tool: {e}")

        # 解析 Memory Manager 结果 (Original Logic)
        def resolve_id(simple_id):
            real = uuid_mapping.get(str(simple_id))
            if not real and not training_mode:
                print(f"   ⚠️ Warning: LLM hallucinated ID '{simple_id}', ignoring.")
            return real

        for tool_call in mem_calls:
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

    # --- Step 4: Execute ---
    def step_execute(self, decisions: List[Dict], extract_result: Dict, user_id: str = 'default'):
        # 使用extract_result中的timestamp和chunk_id
        ts = extract_result['timestamp']
        chunk_id = extract_result['chunk_id']
        all_new_facts = extract_result['new_facts']

        if not decisions:
            # 如果没有决策，确保新事实依然被保存
            if all_new_facts:
                self._save_facts(all_new_facts, ts, chunk_id, user_id)
                # print("   🚫 [User Request] Skipping saving facts to DB.")
            return

        has_non_noop_action = False
        
        for decision in decisions:
            action = decision.get("action")
            if action == "NOOP":
                self.operation_counts["NOOP"] += 1
                continue

            has_non_noop_action = True

            # --- Memory Operations (Case 1-4) ---
            if action == "ADD":
                self.operation_counts["ADD"] += 1
                target_mem_id = str(uuid.uuid4())
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", [], decision.get('user_id', 'default'))
                print(f"   Created Mem: {target_mem_id[:8]}... | Content: {decision['summary']}")

            elif action == "UPDATE":
                self.operation_counts["UPDATE"] += 1
                target_mem_id = decision['target_id']
                
                # 查询旧的memory内容用于打印
                old_memories = self.client.query(
                    collection_name=self.semantic_col,
                    filter=f"memory_id == '{target_mem_id}'",
                    output_fields=["content"]
                )
                old_content = "" if not old_memories else old_memories[0].get("content", "")
                
                self._upsert_mem(target_mem_id, decision['new_content'], decision['orig_created'], ts, "active", [], decision.get('user_id', 'default'))
                print(f"   Updated Mem: {target_mem_id[:8]}...")
                print(f"      Before: {old_content[:]}...")
                print(f"      After:  {decision['new_content'][:]}...")

            elif action == "DELETE":
                self.operation_counts["DELETE"] += 1
                target_mem_id = decision['target_id']
                self._upsert_mem(target_mem_id, "(Archived)", decision['orig_created'], ts, "archived", [], decision.get('user_id', 'default'))
                print(f"   Deleted Mem: {target_mem_id[:8]}...")

            elif action == "INFER":
                self.operation_counts["INFER"] += 1
                target_mem_id = str(uuid.uuid4())
                source_ids = decision.get('source_ids', [])
                
                # 查询 source memories 用于展示
                source_mems = []
                if source_ids:
                    quoted_source_ids = [f'"{sid}"' for sid in source_ids]
                    mem_filter = f"status == 'active' and memory_id in [{','.join(quoted_source_ids)}]"
                    try:
                        source_mems = self.client.query(
                            collection_name=self.semantic_col,
                            filter=mem_filter,
                            output_fields=["content", "memory_id"]
                        )
                    except Exception as e:
                        print(f"   ⚠️ 查询source memory失败: {e}")

                relations = [{"type": "inferred_from", "target_id": sid} for sid in source_ids]
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", relations, decision.get('user_id', 'default'))
                
                # 打印详细的 Infer 过程
                print(f"   💡 Inferred Mem: {target_mem_id[:8]}... | From: {[s[:8] for s in source_ids]}")
                print(f"   ┌─────────────────────────────────────────────────────────────────────────────────")
                if source_mems:
                    print(f"   │ 📋 Infer 前的 Memory ({len(source_mems)}个):")
                    for mem in source_mems:
                        print(f"   │      📌 ID: {mem['memory_id'][:8]}... | 内容: {mem['content'][:]}...")
                print(f"   │ 📝 Infer生成的 Memory:")
                print(f"   │      📌 ID: {target_mem_id[:8]}... | 内容: {decision['summary'][:]}...")
                print(f"   └─────────────────────────────────────────────────────────────────────────────────")

            # --- Fact Operations (Case 5-6) ---
            elif action == "FACT_ADD":
                self.operation_counts["ADD"] += 1 
                print(f"   🆕 Fact Added: {decision['summary']}")

            elif action == "FACT_TRAJECTORIZE":
                self.operation_counts["UPDATE"] += 1
                content = decision['content']
                related_fact_ids = decision.get('related_fact_ids', [])
                
                print(f"   📈 Fact Trajectory: {content[:]}...")
                print(f"      Archiving {len(related_fact_ids)} facts...")

                # 1. Archive old facts
                if related_fact_ids:
                    for fid in related_fact_ids:
                        try:
                            facts = self.client.query(
                                collection_name=self.fact_col,
                                filter=f'fact_id == "{fid}"',
                                output_fields=["fact_id", "details", "text", "timestamp", "user_id"]
                            )
                            if facts:
                                fact = facts[0]
                                details = fact.get('details', [])
                                if isinstance(details, list):
                                    if "Status: Archived" not in details:
                                        details.append("Status: Archived")
                                        details.append(f"Archived Reason: Trajectorized into {content[:]}...")
                                
                                self.client.upsert(self.fact_col, [{
                                    "fact_id": fid,
                                    "linked_chunk_id": fact.get('linked_chunk_id', chunk_id),
                                    "text": fact['text'],
                                    "details": details,
                                    "timestamp": fact['timestamp'],
                                    "user_id": fact.get('user_id', user_id),
                                    "embedding": self._generate_fact_embedding(fact['text'], details)
                                }])
                                print(f"   Archived fact: {fid}")
                        except Exception as e:
                            print(f"Error archiving fact {fid}: {e}")

                # 2. Create new Trajectory Fact
                traj_fact_id = str(uuid.uuid4())
                traj_details = ["Type: Trajectory"]
                self.client.upsert(self.fact_col, [{
                    "fact_id": traj_fact_id,
                    "linked_chunk_id": chunk_id,
                    "text": content,
                    "details": traj_details,
                    "timestamp": ts,
                    "user_id": user_id,
                    "embedding": self._generate_fact_embedding(content, traj_details)
                }])
                print(f"   Created trajectory fact: {traj_fact_id}")

            # --- Core Memory Operations (Case 7) ---

            elif action == "CORE_MEMORY_ADD":
                content = decision['content']
                self.core_memory += f"{content}"
                self._save_core_memory(decision.get('user_id', 'default'))
                print(f"   🧠 Core Memory ADD: {content[:]}...")

            elif action == "CORE_MEMORY_UPDATE":
                old_text = decision['old_text'].strip()
                new_text = decision['new_text'].strip()
                
                # 尝试精确匹配（忽略首尾空格）
                if old_text in self.core_memory:
                    self.core_memory = self.core_memory.replace(old_text, new_text)
                    self._save_core_memory(decision.get('user_id', 'default'))
                    print(f"   🧠 Core Memory UPDATE: {old_text[:]}... -> {new_text[:]}...")
                else:
                    # 尝试模糊匹配：忽略标点符号和空白字符
                    import re
                    def normalize(t):
                        return re.sub(r'[^\w\s]', '', t).strip()
                    
                    normalized_core = normalize(self.core_memory)
                    normalized_old = normalize(old_text)
                    
                    if normalized_old in normalized_core:
                        # 如果能模糊匹配到，尝试在原文本中找到对应的原始文本段
                        # 这里简单处理：如果模糊匹配成功但精确失败，打印提示
                        print(f"   ⚠️ Core Memory Update: Exact match failed, but fuzzy match possible. Please use rewrite if update fails.")
                    
                    print(f"   ⚠️ Core Memory Update Failed: Old text not found.")

            elif action == "CORE_MEMORY_REWRITE":
                new_block = decision['new_block_content']
                self.core_memory = new_block
                self._save_core_memory(decision.get('user_id', 'default'))
                print(f"   🧠 Core Memory REWRITE.")

        # --- Final Step: Save ALL new facts (independent of memories) ---
        if all_new_facts:
            self._save_facts(all_new_facts, ts, chunk_id, user_id)
            # print("   🚫 [User Request] Skipping saving facts to DB.")

    def _save_facts(self, facts: List[Dict], ts: int, chunk_id: str, user_id: str):
        """保存事实到数据库，不进行记忆关联"""
        rows = []
        for fact in facts:
            fact_id = fact.get('fact_id', str(uuid.uuid4()))
            rows.append({
                "fact_id": fact_id,
                "linked_chunk_id": chunk_id,
                "text": fact['text'],
                "details": fact['details'],
                "timestamp": ts,
                "user_id": user_id,
                "embedding": self._generate_fact_embedding(fact['text'], fact['details'])
            })
        if rows:
            self.client.upsert(self.fact_col, rows)
            print(f"   💾 Saved {len(rows)} facts to database (independent).")

    def _upsert_mem(self, mem_id, content, c_at, u_at, status, relations, user_id):
        self.client.upsert(self.semantic_col, [{
            "memory_id": mem_id,
            "embedding": get_embedding(content),
            "content": content,
            "user_id": user_id,
            "status": status,
            "created_at": c_at,
            "updated_at": u_at,
            "relations": relations
        }])

    def step_preprocess_facts(self, extract_result: Dict, user_id: str = 'default') -> Dict:
        """
        预处理提取出的事实，检查是否已存在于数据库中，确保从源头上去重
        
        Args:
            extract_result: 提取结果字典，包含new_facts
            user_id: 用户标识，确保只处理当前用户的事实
            
        Returns:
            更新后的提取结果字典，包含fact_id信息
        """
        new_facts = extract_result['new_facts']
        processed_facts = []
        ts = extract_result['timestamp']
        chunk_id = extract_result['chunk_id']
        
        print(f"🔍 [Preprocess Facts] 检查 {len(new_facts)} 个事实是否已存在...")
        
        # 1. 先对同一批次内的事实进行去重，避免同一批次中重复的事实被处理
        unique_facts_in_batch = []
        seen_fact_keys = set()
        for fact in new_facts:
            # 使用fact_text和details的组合作为唯一标识
            fact_key = f"{fact['text']}::{json.dumps(fact['details'], sort_keys=True)}"
            if fact_key not in seen_fact_keys:
                seen_fact_keys.add(fact_key)
                unique_facts_in_batch.append(fact)
        
        if len(unique_facts_in_batch) < len(new_facts):
            print(f"   ✅ 同一批次内去重 {len(new_facts) - len(unique_facts_in_batch)} 个重复事实")
        
        for fact in unique_facts_in_batch:
            fact_text = fact['text']
            fact_details = fact['details']

            
            # 3. 查询数据库中是否存在相同的fact
            existing_fact = None
            try:
                # 先尝试搜索相关事实，避免全量查询
                # 使用更安全的查询方式，基于text的前缀匹配
                # 只查询text字段包含fact_text关键词的事实
                search_vec = get_embedding(fact_text)
                search_results = self.client.search(
                    self.fact_col, [search_vec], 
                    output_fields=["fact_id", "details", "timestamp", "linked_chunk_id", "text"],
                    limit=20,  # 只查询前20个最相似的事实
                    similarity_threshold=0.8  # 设置相似度阈值，只返回相似度较高的事实
                )
                
                # 处理搜索结果，检查是否有完全匹配的事实
                if search_results and search_results[0]:
                    for hit in search_results[0]:
                        res = hit['entity']
                        res_text = res.get("text", "")
                        res_details = res.get("details", [])
                        # 检查是否是相同的事实，考虑到表述可能略有不同
                        # 1. 完全相同的情况
                        if res_text == fact_text and res_details == fact_details:
                            existing_fact = res
                            break
                        # 2. 核心内容相同但表述略有不同的情况（如有无"User"前缀）
                        stripped_res_text = res_text.lower().replace("user ", "").strip()
                        stripped_fact_text = fact_text.lower().replace("user ", "").strip()
                        if stripped_res_text == stripped_fact_text and res_details == fact_details:
                            existing_fact = res
                            break
            
            except Exception as e:
                print(f"   ⚠️ 查询事实时发生错误: {e}")
            
            if existing_fact:
                # 事实已存在，更新timestamp
                fact_id = existing_fact["fact_id"]
                old_ts = existing_fact["timestamp"]
                
                # 获取现有的linked_chunk_id
                existing_chunk = existing_fact.get("linked_chunk_id", "")
                
                # 更新timestamp和关联信息
                self.client.upsert(self.fact_col, [{
                    "fact_id": fact_id,
                    "linked_chunk_id": existing_chunk,
                    "text": fact_text,
                    "details": fact_details,
                    "timestamp": ts,
                    "user_id": user_id,
                    "embedding": self._generate_fact_embedding(fact_text, fact_details)
                }])
                print(f"   🔄 事实已存在，更新timestamp: {fact_id} (旧: {old_ts}, 新: {ts})")
                
                # 将现有事实添加到processed_facts
                processed_fact = {
                    "text": fact_text,
                    "details": fact_details,
                    "fact_id": fact_id,
                    "timestamp": ts  # 🌟 必须包含 timestamp
                }
                processed_facts.append(processed_fact)
                
                # print(f"   🔄 事实已存在，更新timestamp: {fact_id} (旧: {old_ts}, 新: {ts})")
            else:
                # 事实不存在，生成新的fact_id并保存
                fact_id = str(uuid.uuid4())
                # print(f"   🆕 新事实: {fact_id}")
                
                # 保存新事实到数据库
                self.client.upsert(self.fact_col, [{
                    "fact_id": fact_id,
                    "linked_chunk_id": chunk_id,
                    "text": fact_text,
                    "details": fact_details,
                    "timestamp": ts,
                    "user_id": user_id,
                    "embedding": self._generate_fact_embedding(fact_text, fact_details)
                }])
                print(f"   Add fact: {fact_id}")
                
                processed_fact = {
                    "text": fact_text,
                    "details": fact_details,
                    "fact_id": fact_id,
                    "timestamp": ts  # 🌟 必须包含 timestamp
                }
                
                processed_facts.append(processed_fact)
        
        # 更新提取结果
        extract_result['new_facts'] = processed_facts
        return extract_result
    
    def process(self, text, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None, timestamp: int = None, max_history_turns: int = 5, dataset_question: str = ""):
        res = self.step_extract(text, extract_mode=extract_mode, timestamp=timestamp, max_history_turns=max_history_turns, dataset_question=dataset_question)
        
        # 如果是extract_only模式，提取完直接返回，不进行后续处理（节省Embedding成本）
        if self.extract_only:
            print(f"   🛑 [Extract Only] Skipping preprocessing, retrieval and execution.")
            return

        if not res['new_facts']: return

        
        # 预处理事实，检查是否已存在
        res = self.step_preprocess_facts(res, user_id=user_id)
        
        # 检查预处理后是否还有新事实
        if not res['new_facts']:
            print(f"   ✅ 所有事实都已存在，无需处理")
            return
        
        print(f"   新证据: {res['new_facts']}")
        
        ctx_bundles = self.step_retrieve(res, limit=retrieve_limit, user_id=user_id, similarity_threshold=similarity_threshold)
        decisions = self.step_decide(res, ctx_bundles, user_id=user_id)
        self.step_execute(decisions, res, user_id=user_id)
        
    def process_user_memory_infer(self, line, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None, max_history_turns: int = 5):
        """处理用户记忆会话，支持longmemeval数据集格式"""
        # 加载用户的Core Memory
        self._load_core_memory(user_id)
        
        # 重置操作计数，确保每个用户的计数独立
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        dates = line.get("haystack_dates")
        sessions = line.get("haystack_sessions")
        dataset_question = line.get("question", "")

        for session_id, session in enumerate(sessions):
            date = dates[session_id] + " UTC"
            date_format = "%Y/%m/%d (%a) %H:%M UTC"
            date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
            # 生成timestamp
            timestamp = int(date_string.timestamp())
            
            print(f"处理会话 {session_id + 1}/{len(sessions)}: {dates[session_id]}")
            
            # 直接传递session对象给process方法，而不是转换为文本
            # 使用现有的process方法处理会话消息，传递user_id、similarity_threshold和timestamp
            self.process(session, retrieve_limit=retrieve_limit, extract_mode=extract_mode, user_id=user_id, similarity_threshold=similarity_threshold, timestamp=timestamp, max_history_turns=max_history_turns, dataset_question=dataset_question)
        
        # 返回操作次数统计
        return self.operation_counts
        
    def search_memories(self, query_text, top_k=5, fact_top_k=5, user_id: str = 'default', threshold: float = 0.0, similarity_threshold: float = None, enhanced_search: bool = False, use_fact_retrieval: bool = True):
        """搜索记忆并返回每个记忆关联的topk个事实，并根据关联事实进行rerank
        
        Args:
            query_text: 查询文本
            top_k: 返回的记忆数量上限
            fact_top_k: 每个记忆关联的事实数量上限
            user_id: 用户标识，确保只检索当前用户的记忆
            threshold: 相似度阈值，低于该阈值的记忆将被过滤掉
            similarity_threshold: 向量数据库搜索时的相似度阈值，低于该阈值的记忆将被过滤掉
            enhanced_search: 是否启用增强型搜索模式，启用后会增强rerank逻辑
            use_fact_retrieval: 是否使用事实检索模式，启用后会搜索事实集合并根据关联的memory_id获取更多记忆
        """
        query_vec = get_embedding(query_text)
        
        # 添加调试信息
        filter_expr = f"status == 'active' and user_id == '{user_id}'"
        print(f"   🔍 搜索过滤条件: {filter_expr}, 阈值: {threshold}, 向量搜索阈值: {similarity_threshold}")
        
        # ===========================
        # 1. 搜索记忆集合，获取memoryA
        # ===========================
        mem_res = self.client.search(
            self.semantic_col, [query_vec], filter=filter_expr, limit=top_k,  # 搜索更多记忆，避免遗漏
            output_fields=["content", "memory_id", "created_at", "user_id"],  # 包含user_id字段用于调试
            similarity_threshold=similarity_threshold
        )
        
        # ===========================
        # 2. 搜索事实集合，获取相关事实
        # ===========================
        combined_items = []  # 存储memory和fact及其分数，用于统一排序
        memory_dict = {}  # 临时存储memory对象
        fact_dict = {}  # 临时存储fact对象
        
        # 先处理memoryA
        if mem_res and mem_res[0]:
            for hit in mem_res[0]:
                memory = hit['entity']
                memory_id = memory['memory_id']
                similarity_score = hit['distance']
                # 保存相似度得分
                memory["original_score"] = similarity_score
                memory_dict[memory_id] = memory
                # 将memory添加到combined_items中，用于统一排序
                combined_items.append({
                    "type": "memory",
                    "item": memory,
                    "score": similarity_score,  # 使用相似度作为分数
                    "memory_id": memory_id
                })
        
        # 处理fact
        if use_fact_retrieval:
            # 搜索事实集合
            # 增加过滤条件：status == 'active'，防止检索到被归档或删除的事实
            fact_filter = f"user_id == '{user_id}' and status == 'active'"
            
            fact_res = self.client.search(
                self.fact_col, [query_vec], filter=fact_filter, limit=top_k,  # 搜索更多事实，避免遗漏
                output_fields=["text", "timestamp", "fact_id", "details", "user_id", "embedding", "status"]  # 添加embedding字段
            )
            
            if fact_res and fact_res[0]:
                for hit in fact_res[0]:
                    fact = hit['entity']
                    fact_id = fact['fact_id']
                    # 计算fact与query的内积
                    try:
                        # 直接使用数据库中存储的embedding，而不是重新计算
                        fact_vec = fact.get("embedding")
                        if not fact_vec or not isinstance(fact_vec, list):
                            # 如果没有embedding字段或不是列表，重新计算，使用text和details拼接
                            fact_vec = self._generate_fact_embedding(fact["text"], fact.get("details", []))
                    
                        fact_dot_product = sum(a * b for a, b in zip(query_vec, fact_vec))
                        fact["similarity"] = fact_dot_product
                        fact_dict[fact_id] = fact
                        
                        # 将fact添加到combined_items中，用于统一排序
                        combined_items.append({
                            "type": "fact",
                            "item": fact,
                            "score": fact_dot_product,  # 使用内积作为分数
                            "fact_id": fact_id
                        })
                    except Exception as e:
                        print(f"计算事实相关性失败: {e}")
                        continue
        
        # ===========================
        # 3. 分别对 Memory 和 Fact 进行排序并取 TopK
        # ===========================
        # 分离记忆和事实
        memories_items = [item for item in combined_items if item["type"] == "memory"]
        facts_items = [item for item in combined_items if item["type"] == "fact"]
        
        # 分别按分数降序排序
        memories_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        facts_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 各取 top_k
        top_memories = memories_items[:top_k]
        top_facts = facts_items[:top_k]
        
        # 构造最终结果
        results = []
        
        # 处理 Top 记忆
        for item in top_memories:
            memory = item["item"]
            memory["combined_score"] = item["score"]
            memory["type"] = "memory"
            results.append(memory)
            
        # 处理 Top 事实
        for item in top_facts:
            fact = item["item"]
            results.append({
                "memory_id": fact["fact_id"],
                "content": fact["text"],
                "original_score": item["score"],
                "combined_score": item["score"],
                "details": fact.get("details", []),
                "timestamp": fact.get("timestamp"),
                "created_at": fact.get("timestamp", int(time.time())), # 兼容 response_user 的 created_at
                "type": "fact"
            })
        
        return results
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
                    # model="gemini-3-pro-preview",
                    # model="gpt-4o",
                    model=GENERATION_MODEL,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0,
                )
        
        return response

# ==========================================
# 评估相关函数
# ==========================================
def response_user(line, pipeline, retrieve_limit=20, max_facts_per_memory=3, user_id='default', threshold: float = 0.0, enhanced_search: bool = False):
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
    
    # 加载 Core Memory
    pipeline._load_core_memory(user_id)

    # 搜索记忆，传递user_id、threshold和enhanced_search参数
    retrieved_memories = pipeline.search_memories(question, top_k=enhanced_top_k, user_id=user_id, threshold=threshold, enhanced_search=enhanced_search)
    
    # 确保retrieved_memories不是None
    retrieved_memories = retrieved_memories or []
    
    # 使用统一的格式化方法，包含 Core Memory 和 Retrieved Memories
    memories_str = pipeline._format_context(pipeline.core_memory, retrieved_memories)
    
    # 生成响应
    response = pipeline.generate_response(question, question_date_string, memories_str)
    answer = response.choices[0].message.content
    
    return retrieved_memories, memories_str, answer

def process_and_evaluate_user(line, user_index, infer=True, retrieve_limit: int = 3, extract_mode: str = "whole", vector_db_type="milvus", dataset_name="", max_history_turns: int = 5, extract_only: bool = False, response_only: bool = False):
    """
    封装单个用户的所有处理步骤，以便并行执行。
    返回一个包含所有统计信息的字典。
    """
    try:
        # 为每个用户生成唯一的user_id，确保记忆隔离
        user_id = f"user_{user_index}"
        
        # 为每个用户创建独立的pipeline实例，避免多线程竞争
        # 注意：每个用户的pipeline实例不应该清空数据库，clear_db固定为False
        pipeline = MemoryPipeline(vector_db_type=vector_db_type, clear_db=False, dataset_name=dataset_name, extract_only=extract_only)
        
        # 如果是Response Only模式，直接跳过处理，进行检索和生成
        if response_only:
            print(f"⏩ [User {user_index}] Response Only Mode: Skipping processing, retrieving context directly...")
            retrieved_memories, memories_str, answer = response_user(line, pipeline, retrieve_limit, user_id=user_id)
            
            # 确保retrieved_memories不是None
            retrieved_memories = retrieved_memories or []
            
            return {
                "index": user_index,
                "is_correct": False, # 不进行评估
                "counts": {},
                "question": line.get("question", "N/A"),
                "question_type": line.get("question_type", "unknown"),
                "answer": answer,
                "golden_answer": line.get("answer", "N/A"),
                "retrieved_memories": retrieved_memories,
                "context": memories_str,
            }

        # 处理用户记忆会话，传递user_id、extract_mode和max_history_turns
        memory_counts = pipeline.process_user_memory_infer(line, retrieve_limit=retrieve_limit, extract_mode=extract_mode, user_id=user_id, max_history_turns=max_history_turns)

        
        # 如果是提取模式，直接返回，跳过生成回复
        if extract_only:
            return {
                "index": user_index,
                "is_correct": False, # 不进行评估
                "counts": memory_counts,
                "question": line.get("question", "N/A"),
                "question_type": line.get("question_type", "unknown"),
                "answer": "Extract Only Mode - No Answer Generated",
                "golden_answer": line.get("answer", "N/A"),
                "retrieved_memories": [],
                "context": "Extract Only Mode - No Context",
            }

        # 生成问题响应，传递user_id
        retrieved_memories, memories_str, answer = response_user(line, pipeline, retrieve_limit, user_id=user_id)
        
        # 确保retrieved_memories不是None
        retrieved_memories = retrieved_memories or []
        
        # 获取标准答案和问题类型
        golden_answer = line.get("answer")
        question = line.get("question")
        question_type = line.get("question_type", "unknown")
        
        # 评估答案正确性
        is_correct = lme_grader(llm_client, question, golden_answer, answer, model=GENERATION_MODEL)
        
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
    parser.add_argument("--extract-only", action="store_true", help="仅进行提取，跳过预处理、检索和执行步骤，结果记录在memreader_log.jsonl中")
    parser.add_argument("--response-only", action="store_true", help="仅进行响应生成，跳过提取和记忆更新，直接使用现有数据库和Core Memory")
    args = parser.parse_args()
    
    # 初始化内存管道
    pipeline = MemoryPipeline(vector_db_type=args.vector_db_type, clear_db=args.clear_db, mode='eval' if args.eval else 'test', dataset_name=args.dataset_type, extract_only=args.extract_only)
    
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
                
                # 构建会话结构 - 注意：HotpotQA 是基于 context 的问答，不属于对话，因此将 context 放在 system prompt 中，
                # 将问题作为 user message。这里没有多轮对话，所以是一个单轮 session。
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
                # 提交任务 - 确保参数顺序正确：line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type, args.max_history_turns, args.extract_only, args.response_only
                # 注意：这里clear_db固定为False，只在主函数中执行一次清空操作
                future_to_user = {executor.submit(process_and_evaluate_user, line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type, args.max_history_turns, args.extract_only, args.response_only): (line, idx) for idx, line in enumerate(lines)}
                
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
                future_to_user = {executor.submit(process_and_evaluate_user, line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type, args.max_history_turns, args.extract_only): (line, idx) for idx, line in enumerate(lines)}
                
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
                print(f"用户 {result['index']}: {'✓' if result['is_correct'] else '✗'}")
                print(f"  问题类型: {result.get('question_type', 'unknown')}")
                print(f"  问题: {result['question']}")
                print(f"  上下文: {result['context']}")
                print(f"  回答: {result['answer']}...")
                print(f"  标准答案: {result['golden_answer']}...")
                print(f"  记忆操作: {result['counts']}")
                print()
                
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()