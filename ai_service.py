import os
import json
import logging
import asyncio
import time
import re
from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ddgs import DDGS
from groq import AsyncGroq

load_dotenv()

logger = logging.getLogger("AIService")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Get a free key at https://console.groq.com")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)


class RoleType(str, Enum):
    PRESIDENT = "PRESIDENT"
    MP_ELECTED = "MP_ELECTED"
    MP_NOMINATED = "MP_NOMINATED"
    MCA_ELECTED = "MCA_ELECTED"
    MCA_NOMINATED = "MCA_NOMINATED"
    WOMEN_REP = "WOMEN_REP"
    SENATOR_ELECTED = "SENATOR_ELECTED"
    SENATOR_NOMINATED = "SENATOR_NOMINATED"
    GOVERNOR = "GOVERNOR"


def map_role_to_enum(role_str: str) -> RoleType:
    r = role_str.strip().lower()
    if "president" in r: return RoleType.PRESIDENT
    elif "governor" in r: return RoleType.GOVERNOR
    elif "woman" in r or "women" in r: return RoleType.WOMEN_REP
    elif "senator" in r: return RoleType.SENATOR_NOMINATED if "nominated" in r else RoleType.SENATOR_ELECTED
    elif "mca" in r or "assembly" in r or "ward" in r: return RoleType.MCA_NOMINATED if "nominated" in r else RoleType.MCA_ELECTED
    elif "mp" in r or "parliament" in r or "constituency" in r: return RoleType.MP_NOMINATED if "nominated" in r else RoleType.MP_ELECTED
    return RoleType.MP_ELECTED


def get_constitutional_mandate_text(role: RoleType) -> str:
    mandates = {
        RoleType.PRESIDENT: "Article 131 Mandate: Head of State and Government, directs national executive authority, upholds the Constitution, and guarantees national security and unity.",
        RoleType.GOVERNOR: "Article 179 Mandate: Directs county executive policies, administers county revenue allocations, and manages localized service delivery including health and regional infrastructure.",
        RoleType.SENATOR_ELECTED: "Article 96 Mandate: Protects county interests, debates national revenue sharing formulas, and executes structural oversight over county executive expenditures.",
        RoleType.SENATOR_NOMINATED: "Article 96 & 98 Mandate: Represents targeted vulnerable groups in county affairs and participates in legislative oversight.",
        RoleType.WOMEN_REP: "Article 95 & 100 Mandate: Represents county-wide affirmative action seats in Parliament, administers NG-AAF funds, and sponsors social inclusion legislation.",
        RoleType.MP_ELECTED: "Article 95 Mandate: Enacts national legislation, determines revenue allocation, oversees state organs, and administers NG-CDF allocations.",
        RoleType.MP_NOMINATED: "Article 95 & 97 Mandate: Enacts targeted interest group legislation, participates in plenary debates, and sponsors public petitions.",
        RoleType.MCA_ELECTED: "Article 185 Mandate: Exercises ward-level representation, enacts local county assembly legislation, and monitors county executive budget execution.",
        RoleType.MCA_NOMINATED: "Article 177 & 185 Mandate: Enacts special interest county legislation and exercises executive oversight in assembly committees."
    }
    return mandates.get(role, "Constitutional mandate tied to national policy framework administration and public resource oversight.")


class RubricCategoryScore(BaseModel):
    category_name: str
    weight_percentage: float = Field(..., description="Weight percentage allocated to this category")
    score: float = Field(..., ge=0, le=100, description="Score awarded from 0 to 100 based on verified context")
    analysis: str = Field(..., description="Objective rationale citing verified facts or highlighting missing data")


class AICorePriority(BaseModel):
    id: str = Field(..., description="Two-digit ID, e.g., '01'")
    title: str = Field(..., description="Core priority title based on verified focus areas")


class AILeadershipMatchup(BaseModel):
    vulnerability_index: float = Field(..., ge=0, le=100, description="Vulnerability to challengers based on risk metrics")
    performance_score: float = Field(..., ge=0, le=100, description="Matchup performance score synced with impact rating")


class RepresentativeAIReport(BaseModel):
    representative_name: str
    role: str
    location: str
    overall_score: float = Field(..., ge=0, le=100, description="Weighted total performance score")
    office_mandate: str = Field(..., description="Official constitutional mandate statement")
    rubric_breakdown: List[RubricCategoryScore]
    summary_verdict: str
    key_strengths: List[str]
    areas_for_improvement: List[str]
    
    # Core metric scores dynamically calculated by AI from web facts
    jaba_meter: float = Field(..., ge=0, le=100, description="Talk vs Action score (higher means more rhetoric vs actual completion)")
    impact_rating: float = Field(..., ge=0, le=100, description="Delivery and Performance Score (higher means better track record)")
    risk_radar_index: float = Field(..., ge=0, le=100, description="Risk Level for transparency gaps or missing audits (higher means dirtier record)")
    
    # UI specific fields shown in the frontend dashboard
    action_plan_practicality: float = Field(..., ge=0, le=100)
    unrealistic_promises_risk: float = Field(..., ge=0, le=100)
    core_priorities: List[AICorePriority] = Field(..., description="Top 3 verified core focus areas mapped to UI components")
    leadership_matchup: AILeadershipMatchup = Field(..., description="Challenger matchup viability stats")
    
    # UI Contextual Explanations arrays (Renamed to match Frontend & Sync Pipeline)
    talk_vs_action_justification: List[str] = Field(..., description="3 context points explaining Talk vs Action (Jaba Meter) based on facts")
    developmental_delivery_justification: List[str] = Field(..., description="3 context points justifying physical structures and project execution")
    legislative_delivery_justification: List[str] = Field(..., description="3 context points justifying attendance, committee output, structural policy")
    risk_level_justification: List[str] = Field(..., description="3 context points explaining the risk level, identifying missing audits or discrepancies")


def build_targeted_search_queries(name: str, role: RoleType, location: str) -> List[str]:
    clean_name = re.sub(r'["\']', '', name).strip()
    loc = location if location else "Kenya"
    
    if role == RoleType.PRESIDENT:
        return [
            f'{clean_name} Kenya presidential manifesto delivery 2026',
            f'{clean_name} Kenya Cabinet appointments gender rule Article 10'
        ]
    elif role in [RoleType.MP_ELECTED, RoleType.MP_NOMINATED]:
        return [
            f'{clean_name} MP {loc} Bills sponsored Hansard Kenya',
            f'{clean_name} MP {loc} NG-CDF utilization projects'
        ]
    elif role in [RoleType.MCA_ELECTED, RoleType.MCA_NOMINATED]:
        return [
            f'{clean_name} MCA {loc} County Assembly Bills motions',
            f'{clean_name} MCA {loc} executive oversight development'
        ]
    elif role == RoleType.WOMEN_REP:
        return [
            f'{clean_name} Woman Rep {loc} NG-AAF fund projects impact',
            f'{clean_name} Woman Representative {loc} affirmative action legislation'
        ]
    elif role in [RoleType.SENATOR_ELECTED, RoleType.SENATOR_NOMINATED]:
        return [
            f'{clean_name} Senator {loc} county revenue allocation voting record',
            f'{clean_name} Senator {loc} CPAIC county audit oversight Auditor General'
        ]
    elif role == RoleType.GOVERNOR:
        return [
            f'{clean_name} Governor {loc} CIDP development projects delivery',
            f'{clean_name} Governor {loc} Auditor General financial report'
        ]
        
    return [f'{clean_name} {loc} Kenya leadership development news']


def get_constitutional_rubric_prompt(role: RoleType) -> str:
    rubrics = {
        RoleType.PRESIDENT: """
        Evaluate against Article 131 & 132 (Weight total: 100%):
        1. Manifesto & Policy Delivery Rate (35%)
        2. State of the Nation & Article 10 Compliance (25%)
        3. Cabinet & Public Appointment Inclusivity (20%)
        4. Legislative Assent & Veto Management (20%)
        """,
        RoleType.MP_ELECTED: """
        Evaluate against Article 95 (Weight total: 100%):
        1. Legislative Output - Bills & Motions Sponsored (30%)
        2. Hansard Plenary & Committee Participation Rate (25%)
        3. NG-CDF Fund Utilization & Completion (30%)
        4. Public Petitions Tabled (15%)
        """,
        RoleType.MP_NOMINATED: """
        Evaluate against Article 95 & 97 (Weight total: 100%):
        1. Target Group Legislative Interventions - Youth/PWD/Workers (40%)
        2. Hansard Plenary & Committee Participation Rate (30%)
        3. Legislative Output - Bills & Amendments (20%)
        4. Public Advocacy & Petitions (10%)
        """,
        RoleType.MCA_ELECTED: """
        Evaluate against Article 185 (Weight total: 100%):
        1. County Legislation Output - Motions & County Bills (30%)
        2. County Executive Oversight & CEC Questioning (30%)
        3. Ward Infrastructure & Local Development Delivery (25%)
        4. Ward Public Participation Leadership (15%)
        """,
        RoleType.MCA_NOMINATED: """
        Evaluate against Article 177 (Weight total: 100%):
        1. Special Interest Legislation - Gender, Youth, PWD Rights (45%)
        2. County Executive Oversight Participation (30%)
        3. County Assembly Motions & Committee Output (25%)
        """,
        RoleType.WOMEN_REP: """
        Evaluate against Article 97 (Weight total: 100%):
        1. NG-AAF Fund Management & Community Impact (35%)
        2. Affirmative Action Legislation & Advocacy (30%)
        3. National Assembly Plenary & Committee Participation (20%)
        4. Gender-Based Violence & Maternal Health Initiatives (15%)
        """,
        RoleType.SENATOR_ELECTED: """
        Evaluate against Article 96 (Weight total: 100%):
        1. County Revenue Allocation (CARA) Advocacy & Voting Record (35%)
        2. County Audit Oversight & CPAIC Committee Activity (30%)
        3. Devolution-Strengthening Bills & Dispute Interventions (20%)
        4. Plenary Attendance & Hansard Participation (15%)
        """,
        RoleType.SENATOR_NOMINATED: """
        Evaluate against Article 98 (Weight total: 100%):
        1. Vulnerable Group Representation - Women/Youth/PWD Rights (40%)
        2. Devolution Oversight & CPAIC Participation (30%)
        3. Legislative Output & Sponsored Amendments (20%)
        4. Plenary Participation Rate (10%)
        """,
        RoleType.GOVERNOR: """
        Evaluate against Article 179 & 183 (Weight total: 100%):
        1. CIDP Project Implementation Rate (35%)
        2. Audit & Financial Probity - Auditor General Reports (25%)
        3. Essential Service Delivery - Health, ECDE, Roads, Water (25%)
        4. County Executive Inclusivity & Gender Rule Compliance (15%)
        """
    }
    return rubrics.get(role, "Evaluate based on general public service output and Article 10 principles.")


def _execute_duckduckgo_search(name: str, role_str: str, location: str, queries: List[str]) -> str:
    combined_snippets = []
    clean_name = re.sub(r'["\']', '', name).strip()
    
    try:
        with DDGS() as ddgs:
            for q in queries:
                time.sleep(1)
                try:
                    results = list(ddgs.text(q, region='wt-wt', safesearch='moderate', max_results=3))
                    for r in results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        if body:
                            combined_snippets.append(f"Source [{title}]: {body}")
                except Exception as query_err:
                    logger.debug(f"Query fail for '{q}': {query_err}")

            if not combined_snippets:
                fallback_query = f"{clean_name} {location} Kenya {role_str}"
                time.sleep(1)
                try:
                    results = list(ddgs.text(fallback_query, region='wt-wt', safesearch='moderate', max_results=5))
                    for r in results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        if body:
                            combined_snippets.append(f"Source [{title}]: {body}")
                except Exception as fallback_err:
                    logger.debug(f"Fallback fail for '{fallback_query}': {fallback_err}")
                        
    except Exception as e:
        logger.warning(f"DuckDuckGo search error for {name}: {e}")
        
    if combined_snippets:
        return "\n".join(combined_snippets)
    
    return f"CRITICAL: No public web records, news, or development footprint found for {clean_name} serving as {role_str} in {location}. You MUST score them highly on Risk Level (e.g. >75), strictly low on Delivery (e.g. <30), and explicitly document that 'No verifiable public footprint or development records could be found' in your explanations. DO NOT hallucinate achievements."


async def generate_true_ai_deep_dive(
    name: str,
    role: str,
    county: str,
    jaba: float = 15.0,
    impact: float = 70.0,
    rvs: float = 10.0
) -> Dict[str, Any]:
    
    role_enum = map_role_to_enum(role)
    location_str = county or "Kenya"
    mandate_text = get_constitutional_mandate_text(role_enum)
    rubric_prompt = get_constitutional_rubric_prompt(role_enum)
    targeted_queries = build_targeted_search_queries(name, role_enum, location_str)
    
    search_context = await asyncio.to_thread(_execute_duckduckgo_search, name, role, location_str, targeted_queries)
    
    system_instruction = f"""
    You are 'Facts Tupu', an objective Kenyan political scoring engine.
    Evaluate candidate performance strictly based on the Constitution of Kenya (2010).

    Target: {name}
    Role: {role} ({role_enum.value})
    Location Context: {location_str}
    Mandate Statement: "{mandate_text}"
    
    CONSTITUTIONAL RUBRIC:
    {rubric_prompt}

    RULES FOR SCORING:
    1. Evaluate performance strictly from 0 to 100 for each category based ON RETRIEVED WEB CONTEXT.
    2. NEVER invent generic reports, generic focus areas, or unverified achievements. Use ACTUAL FACTS from the web context.
    3. Generate the following dynamic UI metrics:
       - 'jaba_meter' (Talk vs Action %): Calculate based on rhetoric versus actual completion. 
       - 'impact_rating' (Delivery Score %): Evaluate actual verified development track.
       - 'risk_radar_index' (Risk Level %): Evaluate management discrepancies, transparency, missing records.
    4. Provide exactly 3 concise points for each context array (talk_vs_action_justification, developmental_delivery_justification, legislative_delivery_justification, risk_level_justification).
    5. Return exactly 3 verified 'core_priorities' formatted as [{{ "id": "01", "title": "..." }}].
    6. Return 'leadership_matchup' object with 'vulnerability_index' and 'performance_score'.
    7. Return your response in strict JSON format matching the exact schema keys requested.
    
    STRICT JSON SCHEMA KEYS TO INCLUDE:
    "representative_name", "role", "location", "office_mandate", "overall_score", 
    "rubric_breakdown", "summary_verdict", "key_strengths", "areas_for_improvement", 
    "jaba_meter", "impact_rating", "risk_radar_index", "action_plan_practicality", 
    "unrealistic_promises_risk", "core_priorities", "leadership_matchup", 
    "talk_vs_action_justification", "developmental_delivery_justification", 
    "legislative_delivery_justification", "risk_level_justification"
    """

    user_prompt = f"""
    Retrieved Web Context:
    {search_context}
    
    Current Database Baseline Scores (Use only as reference if context is ambiguous):
    - Jaba Meter (Talk vs Action): {jaba}%
    - Impact Rating (Delivery Score): {impact}%
    - Risk Radar Index: {rvs}%
    
    Evaluate candidate {name} ({role}, {location_str}) using ONLY the context above. If the context reveals actual facts, override the baseline scores completely with your fact-based calculation. If context is completely empty, heavily penalize the Delivery score and increase Risk. Provide specific explanations matching the required UI fields.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            ai_output_str = response.choices[0].message.content
            parsed_data = json.loads(ai_output_str) # pyright: ignore[reportArgumentType]
            
            validated_model = RepresentativeAIReport(**parsed_data)
            final_dict = validated_model.model_dump()
            
            await asyncio.sleep(2)
            return final_dict
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                logger.warning(f"Groq rate limit hit for {name}. Retrying in 12s...")
                await asyncio.sleep(12)
                continue
            else:
                logger.error(f"Error generating AI deep dive for {name}: {e}")
                break

    # Structured fallback ensuring UI components receive critical baseline strings matching the exact schema
    calculated_score = max(50.0, float(impact))
    return {
        "representative_name": name,
        "role": role,
        "location": location_str,
        "overall_score": calculated_score,
        "office_mandate": mandate_text,
        "rubric_breakdown": [
            {
                "category_name": "Constitutional Mandate & Oversight",
                "weight_percentage": 50.0,
                "score": calculated_score,
                "analysis": "No verifiable public data available to confirm output."
            },
            {
                "category_name": "Development & Project Delivery",
                "weight_percentage": 50.0,
                "score": calculated_score - 2.0,
                "analysis": "Evaluation constrained by lack of internet footprint."
            }
        ],
        "summary_verdict": f"Limited verifiable data available for {name}. Assessed using baseline indicators.",
        "key_strengths": ["Active Public Office Holder"],
        "areas_for_improvement": ["Public Accessibility", "Data Transparency"],
        "jaba_meter": jaba,
        "impact_rating": impact,
        "risk_radar_index": rvs,
        "action_plan_practicality": 35.0,
        "unrealistic_promises_risk": 68.0,
        "core_priorities": [
            {"id": "01", "title": "Public Record Transparency"}, 
            {"id": "02", "title": "Constitutional Mandate Verification"}, 
            {"id": "03", "title": "Local Development Accountability"}
        ],
        "leadership_matchup": {
            "vulnerability_index": min(rvs * 2, 90.0),
            "performance_score": impact
        },
        "talk_vs_action_justification": [
            "Insufficient public platform footprint to measure rhetoric.",
            "Pending ground verification of promises.",
            "Information severely constrained."
        ],
        "developmental_delivery_justification": [
            "No verifiable physical projects logged in recent news.",
            "Project funding execution lacks public trails.",
            "Evaluation defaulted to baseline due to low data."
        ],
        "legislative_delivery_justification": [
            "Statutory review tracking unavailable.",
            "Committee output lacks accessible digitization.",
            "Pending Hansard review verification."
        ],
        "risk_level_justification": [
            "High vulnerability triggered by missing public audit trails.",
            "Transparency significantly lags recommended baseline.",
            "Discrepancies cannot be ruled out without records."
        ]
    }