from utils import LME_JUDGE_MODEL_TEMPLATE, extract_json
import json
from pydantic import BaseModel, Field

class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


def lme_grader(llm_client, question, golden_answer, response, model="gpt-4o-mini"):
    system_prompt = """You are an expert grader that determines if answers to questions match a gold standard answer"""
    judge_prompt = LME_JUDGE_MODEL_TEMPLATE.format(
        question=question, golden_answer=golden_answer, response=response
    )

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )

        message_content = response.choices[0].message.content
        
        try:
            # Try to extract and parse JSON
            json_content = extract_json(message_content)
            label = json.loads(json_content)["label"]
        except Exception:
            # Fallback: check for keywords if JSON parsing fails
            content_upper = message_content.upper()
            if "CORRECT" in content_upper and "WRONG" not in content_upper:
                label = "CORRECT"
            elif "WRONG" in content_upper and "CORRECT" not in content_upper:
                label = "WRONG"
            elif "CORRECT" in content_upper: # If both present, maybe prioritize correct if it seems to be the label? 
                # Actually, usually "CORRECT" appears in reasoning too.
                # Let's try to be smarter. 
                # If structure failed, maybe it's just the word "CORRECT" or "WRONG"
                if message_content.strip().upper() == "CORRECT":
                    label = "CORRECT"
                elif message_content.strip().upper() == "WRONG":
                    label = "WRONG"
                else:
                    # Last resort: simple substring check if unambiguous
                    if "label" in message_content and "CORRECT" in message_content:
                        label = "CORRECT"
                    elif "label" in message_content and "WRONG" in message_content:
                        label = "WRONG"
                    else:
                        raise ValueError(f"Could not extract label from: {message_content}")

        parsed = LLMGrade(llm_judgment=label, llm_reasoning="")

        return parsed.llm_judgment.strip().lower() == "correct"
    except Exception as e:
        print(f"评估答案正确性时出错: {e}")
        # print(f"Raw content: {message_content}") # Debug
        return False