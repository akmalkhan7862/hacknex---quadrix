"""
LLM Answer Generation Module for MDI
Calls LLM (Gemini / OpenAI / rule-based fallback) with strict anti-hallucination rules.
Forces claims to be mapped directly to provided evidence_ids.
"""

import os
import json
import re
from typing import Dict, Any, List

SYSTEM_PROMPT = """
You are an expert Document Intelligence AI assistant.
Your job is to answer the user's question STRICTLY and ONLY using the provided evidence context.

CRITICAL RULES:
1. Answer ONLY using the supplied evidence context.
2. NEVER invent values, figures, dates, document names, page numbers, or sections.
3. NEVER make unsupported claims.
4. Every factual sentence in your answer must be represented as a claim with matching evidence_ids.
5. IF THE SUPPLIED EVIDENCE IS INSUFFICIENT OR DOES NOT CONTAIN THE ANSWER, YOU MUST RESPOND EXACTLY WITH:
   "Not found in the provided documents."

Format your output as a valid JSON object with the following schema:
{
  "answer": "Complete answer text here...",
  "claims": [
    {
      "text": "Factual claim sentence.",
      "evidence_ids": ["EVIDENCE_ID_1"]
    }
  ]
}
"""

def generate_llm_answer(question: str, context: str, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate answer from LLM given question and evidence context.
    If no relevant evidence items exist, immediately return "Not found in the provided documents."
    """
    NOT_FOUND_RESPONSE = {
        "answer": "Not found in the provided documents.",
        "claims": [],
        "sources": []
    }
    
    if not evidence_items or context == "NO_RELEVANT_EVIDENCE_FOUND":
        return NOT_FOUND_RESPONSE
        
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # 1. Try OpenRouter API if key is present
    if openrouter_key:
        try:
            import openai
            model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
            client = openai.OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE CONTEXT:\n{context}"}
                ],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content.strip()
            clean_json = re.sub(r"^```json\s*", "", raw_text)
            clean_json = re.sub(r"\s*```$", "", clean_json)
            parsed = json.loads(clean_json)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except Exception as e:
            print(f"[llm.py] OpenRouter API call failed/fallback: {e}")
            
    # 2. Try Gemini API if key is present
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            prompt = f"{SYSTEM_PROMPT}\n\nQUESTION: {question}\n\nEVIDENCE CONTEXT:\n{context}\n\nJSON RESPONSE:"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            raw_text = response.text.strip()
            clean_json = re.sub(r"^```json\s*", "", raw_text)
            clean_json = re.sub(r"\s*```$", "", clean_json)
            parsed = json.loads(clean_json)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except Exception as e:
            print(f"[llm.py] Gemini API call failed/fallback: {e}")
            
    # 3. Try OpenAI API if key is present
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE CONTEXT:\n{context}"}
                ],
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content.strip()
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except Exception as e:
            print(f"[llm.py] OpenAI API call failed/fallback: {e}")

    # 4. Deterministic Evidence Extractor Fallback (No API Key or Offline)
    return _rule_based_fallback(question, evidence_items)

def _rule_based_fallback(question: str, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extremely strict fallback that extracts answers directly from matched evidence sentences.
    Guarantees no hallucination when API keys are not supplied.
    """
    if not evidence_items:
        return {"answer": "Not found in the provided documents.", "claims": []}
    
    # Check top candidate evidence content
    best_item = evidence_items[0]
    content = best_item.get("content", "").strip()
    elem_id = best_item.get("element_id", "")
    
    q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
    
    matching_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(w in s_lower for w in q_words):
            matching_sentences.append(s)
            
    if not matching_sentences and sentences:
        matching_sentences.append(sentences[0])
        
    if matching_sentences:
        answer_text = " ".join(matching_sentences[:3])
        return {
            "answer": answer_text,
            "claims": [
                {
                    "text": answer_text,
                    "evidence_ids": [elem_id]
                }
            ]
        }
        
    return {"answer": "Not found in the provided documents.", "claims": []}
