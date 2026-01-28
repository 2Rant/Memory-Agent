from utils import LME_JUDGE_MODEL_TEMPLATE
import json
from pydantic import BaseModel, Field

class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


def lme_grader(llm_client, question, golden_answer, response):
    system_prompt = """You are an expert grader that determines if answers to questions match a gold standard answer"""
    judge_prompt = LME_JUDGE_MODEL_TEMPLATE.format(
        question=question, golden_answer=golden_answer, response=response
    )

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )

        message_content = response.choices[0].message.content
        label = json.loads(message_content)["label"]
        parsed = LLMGrade(llm_judgment=label, llm_reasoning="")

        return parsed.llm_judgment.strip().lower() == "correct"
    except Exception as e:
        print(f"评估答案正确性时出错: {e}")
        return False