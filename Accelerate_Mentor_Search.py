import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import os
import json
import re
import hashlib
from datetime import datetime
from io import BytesIO
import pdfplumber
import docx

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resources Network – Mentor Search",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────────────────────
#  PASSWORD PROTECTION
# ─────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    st.markdown("### 🔒 Please enter the password to continue")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login"):
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    return False

if not check_password():
    st.stop()

# ─────────────────────────────────────────────
#  CLIENT
# ─────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ─────────────────────────────────────────────
#  DOCUMENT EXTRACTION UTILS
# ─────────────────────────────────────────────
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass
    return text.strip()

def extract_text_from_docx_bytes(file_bytes):
    text = ""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception:
        pass
    return text.strip()

def extract_text_from_file_path(file_path):
    if not file_path or not isinstance(file_path, str) or file_path.strip() == "":
        return ""
    fp = file_path.strip()
    try:
        if fp.lower().endswith(".pdf"):
            with open(fp, "rb") as f:
                return extract_text_from_pdf_bytes(f.read())
        elif fp.lower().endswith(".docx"):
            with open(fp, "rb") as f:
                return extract_text_from_docx_bytes(f.read())
        elif fp.lower().endswith(".txt"):
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""
    return ""

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
_INDUSTRY_LABELS = [
    "Agri / Food Processing", "Manufacturing", "Healthcare", "Climate Tech",
    "Deep Tech", "Enterprise Tech", "D2C / B2C", "Services",
    "Fintech & Financial Services", "Automotive & Auto Components",
    "Media & Entertainment", "HR Services", "Legal Services",
    "Transportation & Logistics", "Other (please specify)",
]

def parse_operator_industries(val):
    parts = [p.strip() for p in str(val).split("|")]
    active = [
        _INDUSTRY_LABELS[i]
        for i, p in enumerate(parts)
        if p and p.lower() not in ("", "nan", "0", "no", "false") and i < len(_INDUSTRY_LABELS)
    ]
    return ", ".join(active) if active else ""

@st.cache_data
def load_data():
    df = pd.read_excel("mentors.xlsx", engine="openpyxl", dtype=str, na_filter=False)

    required_cols = [
        "Name", "LinkedIn", "Primary Expertise", "Secondary Expertise",
        "Primary Industry", "Secondary Industry", "Description",
        "Document Path",
        "Program", "Years of Experience", "Location",
        "Current Organization", "Current Designation", "Qualification",
        "Industry - Operator Data",
        "What is the one business problem you are most qualified to advise on from direct experience?",
        "Other Experience(s), if any",
        "What revenue stage do you understand best from the inside? (Select one only)",
        "Describe one time you helped a business break through a growth ceiling.* (What was the ceiling, and what specifically changed?)",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    is_cloud = (
        os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"
        or os.environ.get("IS_STREAMLIT_CLOUD") is not None
        or not os.path.exists(os.path.expanduser("~/.streamlit/config.toml"))
    )
    df["Doc Text"] = "" if is_cloud else df["Document Path"].apply(extract_text_from_file_path)
    df["Active Industries"] = df["Industry - Operator Data"].apply(parse_operator_industries)

    def safe(name):
        return df[name] if name in df.columns else ""

    df["combined"] = (
        "Primary Expertise: " + safe("Primary Expertise") + ". " +
        "Secondary Expertise: " + safe("Secondary Expertise") + ". " +
        "Primary Industry: " + safe("Primary Industry") + ". " +
        "Secondary Industry: " + safe("Secondary Industry") + ". " +
        "Active Industry Sectors: " + safe("Active Industries") + ". " +
        "Description: " + safe("Description") + ". " +
        "Qualification: " + safe("Qualification") + ". " +
        "Current Organization: " + safe("Current Organization") + ". " +
        "Current Designation: " + safe("Current Designation") + ". " +
        "Core Business Problem Advised: " + safe("What is the one business problem you are most qualified to advise on from direct experience?") + ". " +
        "Other Experiences: " + safe("Other Experience(s), if any") + ". " +
        "Revenue Stage Expertise: " + safe("What revenue stage do you understand best from the inside? (Select one only)") + ". " +
        "Growth Ceiling Story: " + safe("Describe one time you helped a business break through a growth ceiling.* (What was the ceiling, and what specifically changed?)")
    )
    df["combined"] = df["combined"].fillna("").astype(str).str.strip()
    return df

df = load_data()

# Lookup dicts
program_lookup    = df.set_index("Name")["Program"].to_dict()            if "Program"            in df.columns else {}
experience_lookup = df.set_index("Name")["Years of Experience"].to_dict() if "Years of Experience" in df.columns else {}
linkedin_lookup   = df.set_index("Name")["LinkedIn"].to_dict()            if "LinkedIn"           in df.columns else {}

# ─────────────────────────────────────────────
#  SENTENCE TRANSFORMER + VECTORS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def get_vectors(texts):
    clean = [str(t).strip() if t and str(t).strip() else "no information available" for t in texts]
    return model.encode(clean, batch_size=64, show_progress_bar=False)

vectors = get_vectors(df["combined"].tolist())

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
hc1, hc2, hc3 = st.columns([1, 3, 1])
with hc1:
    try:
        st.image("DP_BG1.png", width=150)
    except Exception:
        pass
with hc2:
    st.markdown(
        "<h2 style='text-align:center;'>🌐 Resources Network – Mentor Search</h2>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# Program filter (shared across both tabs)
_all_programs = sorted(set(
    p.strip()
    for val in df["Program"]
    for p in str(val).split(",")
    if p.strip() and p.strip().lower() not in ("nan", "none", "")
))

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filter by Program")
st.sidebar.caption("Applies to both tabs.")
selected_programs = st.sidebar.multiselect(
    "Program(s)", options=_all_programs, default=[],
    key="program_filter", placeholder="All programs"
)

if selected_programs:
    _prog_mask = df["Program"].apply(
        lambda x: any(
            sel.strip() in [p.strip() for p in str(x).split(",")]
            for sel in selected_programs
        )
    )
    filtered_df = df[_prog_mask].reset_index(drop=True)
    filtered_vectors = get_vectors(filtered_df["combined"].tolist())
    st.sidebar.info(f"🔎 **{len(filtered_df)}** mentor(s) in: {', '.join(selected_programs)}")
else:
    filtered_df = df
    filtered_vectors = vectors
    st.sidebar.caption(f"🔎 Searching all **{len(df)}** mentors")

# Clear chat
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat & History"):
    for k in ["messages", "last_recommendations", "last_query",
              "pending_retry", "retry_query", "search_history"]:
        st.session_state[k] = [] if k in ("messages", "last_recommendations", "search_history") else ""
    st.session_state["pending_retry"] = False
    st.rerun()

# Recent searches
st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Recent Searches")
if not st.session_state.get("search_history"):
    st.sidebar.caption("No searches yet.")
else:
    for i, item in enumerate(reversed(st.session_state.search_history[-10:])):
        label = item["query"][:40] + ("…" if len(item["query"]) > 40 else "")
        meta  = f"🏆{item['tier1']} · 🔍{item['tier2']}  {item['timestamp']}"
        if st.sidebar.button(f"↩ {label}", key=f"sidebar_rerun_{i}",
                             use_container_width=True, help=item["query"]):
            st.session_state._rerun_query = item["query"]
            st.rerun()
        st.sidebar.caption(meta)

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "messages": [],
    "last_recommendations": [],
    "last_query": "",
    "pending_retry": False,
    "retry_query": "",
    "search_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  HELPERS – SEARCH TAB
# ─────────────────────────────────────────────
def save_to_history(query, t1, t2):
    st.session_state.search_history = [
        h for h in st.session_state.search_history
        if h["query"].strip().lower() != query.strip().lower()
    ]
    st.session_state.search_history.append({
        "query": query,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "tier1": t1, "tier2": t2,
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[-10:]

_SEARCH_STOPWORDS = {
    "looking", "experts", "expert", "need", "help", "want", "mentor",
    "mentors", "industry", "with", "from", "that", "have", "for", "the",
    "and", "who", "can", "are", "best", "good", "find", "show", "give",
    "someone", "person", "people", "startup", "business", "company",
    "founder", "their", "this", "about", "know", "does", "has"
}

def extract_keywords(query):
    words = re.findall(r'\b[\w&/]+\b', query.lower())
    return [w for w in words if len(w) > 2 and w not in _SEARCH_STOPWORDS]

def detect_intent(user_input, last_recommendations):
    followup_keywords = [
        "tell me more", "more about", "compare", "vs", "versus",
        "which is better", "difference between", "what about",
        "explain", "elaborate", "details about", "why is", "how is",
        "refine", "show more", "different", "another", "instead",
        "same industry", "similar", "like #", "expert #", "first expert",
        "second expert", "third expert", "top expert", "number"
    ]
    input_lower = user_input.lower()
    is_followup = bool(last_recommendations) and any(kw in input_lower for kw in followup_keywords)
    return "followup" if is_followup else "new_search"

def call_ai(prompt, max_tokens=2048):
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        temperature=0,
        system="""You are an AI expert-matching assistant for Resources Network,
helping Indian founders find the most suitable experts from a curated database.

CORE RULES:
- Industry match alone is NOT enough for Tier 1
- Hands-on operator experience alone is NOT enough for Tier 1
- BOTH must be present for Tier 1
- When in doubt → Tier 2, not Tier 1
- Never assume industry match if not clearly stated in the profile
- Be honest — do not mark Yes for hands-on just because the expert is impressive

MINIMUM RESULT RULES:
- Always return at least 1 result total
- If no Tier 1 experts exist, return the best available as Tier 2
- Never return an empty array""",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def render_program_badge(expert_name):
    prog_val = program_lookup.get(expert_name, "").strip()
    if not prog_val or prog_val.lower() in ("", "nan", "none"):
        return
    prog_list = [p.strip() for p in prog_val.split(",") if p.strip()]
    if not prog_list:
        return
    badges_html = " ".join([
        f"<span style='background:#1F4E79;color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;margin-right:4px;font-weight:500;'>"
        f"📌 {p}</span>" for p in prog_list
    ])
    st.markdown(
        f"<div style='margin-bottom:8px;'>"
        f"<span style='font-size:13px;color:#555;font-weight:500;'>Program: </span>"
        f"{badges_html}</div>", unsafe_allow_html=True
    )

def render_experience_badge(expert_name):
    exp_val = experience_lookup.get(expert_name, "").strip()
    if not exp_val or exp_val.lower() in ("", "nan", "none"):
        return
    st.markdown(
        f"<div style='margin-bottom:12px;'>"
        f"<span style='font-size:13px;color:#555;font-weight:500;'>Experience: </span>"
        f"<span style='background:#E2EFDA;color:#375623;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:500;'>"
        f"🏅 {exp_val}</span></div>", unsafe_allow_html=True
    )

def display_expert_card(expert, index, source_df):
    hands_on = expert.get("Hands On Experience", "").strip()
    badge = {"Yes": "🟢 Hands-On/Operator",
             "Partial": "🟡 Partial Hands-on/Operator"}.get(hands_on, "🔴 No Direct Experience")

    overall     = expert.get("Overall Score", "N/A")
    expert_name = expert.get("Name", "N/A")

    with st.expander(f"#{index} — {expert_name} | ⭐ {overall}/10 | {badge}", expanded=(index == 1)):
        render_program_badge(expert_name)
        render_experience_badge(expert_name)

        st.markdown("### 📊 Match Scorecard")
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, key, label, max_val in [
            (sc1, "Industry Match Score",  "🏭 Industry Match",          "3"),
            (sc2, "Hands On Score",        "🛠️ Hands-on/Operator Exp",   "3"),
            (sc3, "Expertise Score",       "💼 Expertise",               "2"),
            (sc4, "Credibility Score",     "🏅 Key Credentials",         "2"),
        ]:
            with col:
                raw   = expert.get(key, "N/A")
                parts = raw.split("|") if isinstance(raw, str) and "|" in raw else [raw, ""]
                st.metric(label, f"{parts[0].strip()} / {max_val}")
                if len(parts) > 1:
                    st.caption(parts[1].strip())

        st.markdown("---")
        st.markdown("**🎯 Core Area of Expertise**")
        st.write(expert.get("Core Expertise", "Not available"))

        st.markdown("**✅ Why Suitable**")
        st.write(expert.get("Match Reason", ""))

        st.markdown("**💼 Relevant Experience**")
        st.write(expert.get("Relevant Experience", ""))

        st.markdown("**🛠️ Hands-on/Operator Experience**")
        if hands_on == "Yes":
            st.success(f"✅ Yes — {expert.get('Hands On Details', '')}")
        elif hands_on == "Partial":
            st.warning(f"⚠️ Partial — {expert.get('Hands On Details', '')}")
        else:
            st.error(f"❌ No — {expert.get('Hands On Details', '')}")

        # Problem & Stage Fit (shown only if non-empty)
        cp = expert.get("Core Problem Match", "").strip()
        rf = expert.get("Revenue Stage Fit", "").strip()
        gc = expert.get("Growth Ceiling Relevance", "").strip()
        if any(v and v.lower() not in ("not specified", "n/a", "") for v in [cp, rf, gc]):
            st.markdown("---")
            st.markdown("**🎯 Problem & Stage Fit**")
            ca, cb, cc = st.columns(3)
            with ca:
                if cp and cp.lower() not in ("not specified", "n/a", ""):
                    st.markdown("**🔑 Core Problem Match**"); st.caption(cp)
            with cb:
                if rf and rf.lower() not in ("not specified", "n/a", ""):
                    st.markdown("**📈 Revenue Stage Fit**"); st.caption(rf)
            with cc:
                if gc and gc.lower() not in ("not specified", "n/a", ""):
                    st.markdown("**🚀 Growth Ceiling Relevance**"); st.caption(gc)

        linkedin = linkedin_lookup.get(expert_name, "")
        if linkedin and str(linkedin).strip():
            st.markdown(f"[🔗 View LinkedIn Profile]({linkedin})")

def display_expert_results(ai_recommendations, source_df):
    if not ai_recommendations:
        return
    tier1 = [m for m in ai_recommendations if m.get("Tier") == "1"]
    tier2 = [m for m in ai_recommendations if m.get("Tier") == "2"]

    if tier1:
        st.markdown("## 🏆 Tier 1 — Strong Matches\n"
                    "> Match **both industry AND hands-on operator experience**.")
        for i, e in enumerate(tier1):
            display_expert_card(e, i + 1, source_df)
    else:
        st.warning("⚠️ No Tier 1 matches — no expert matched both industry AND operator experience.")

    if tier2:
        st.markdown("---\n## 🔍 Tier 2 — Partial Matches\n"
                    "> Match **either industry OR relevant experience**, but not both.")
        for i, e in enumerate(tier2):
            display_expert_card(e, i + 1, source_df)

def run_search(query, source_df, source_vectors):
    query_vec  = model.encode([query])
    similarity = cosine_similarity(query_vec, source_vectors)
    source_df  = source_df.copy()
    source_df["semantic_score"] = similarity[0]

    keywords = extract_keywords(query)
    if keywords:
        source_df["keyword_boost"] = source_df["combined"].apply(
            lambda t: sum(0.08 for kw in keywords if kw in str(t).lower())
        )
    else:
        source_df["keyword_boost"] = 0.0

    source_df["score"] = source_df["semantic_score"] + source_df["keyword_boost"]
    candidates = source_df.sort_values(by=["score", "Name"], ascending=[False, True]).head(30)

    expert_info = ""
    for _, row in candidates.iterrows():
        doc_summary   = row["Doc Text"][:500] if row["Doc Text"] and len(row["Doc Text"]) > 50 else "Not available"
        active_sectors = row.get("Active Industries", "").strip()
        core_problem   = row.get("What is the one business problem you are most qualified to advise on from direct experience?", "").strip()
        other_exp      = row.get("Other Experience(s), if any", "").strip()
        revenue_stage  = row.get("What revenue stage do you understand best from the inside? (Select one only)", "").strip()
        growth_story   = row.get("Describe one time you helped a business break through a growth ceiling.* (What was the ceiling, and what specifically changed?)", "").strip()

        def r(c):
            v = row.get(c, "")
            return str(v).strip() if v and str(v).strip().lower() not in ("nan", "0") else ""

        expert_info += f"""
Name: {row['Name']}
Primary Expertise: {r('Primary Expertise')}
Secondary Expertise: {r('Secondary Expertise')}
Primary Industry: {r('Primary Industry')}
Active Industry Sectors: {active_sectors or 'Not specified'}
Current Designation: {r('Current Designation')}
Current Organization: {r('Current Organization')}
Qualification: {r('Qualification')}
Description: {r('Description')}
Core Business Problem They Can Advise On: {core_problem or 'Not specified'}
Other Experiences: {other_exp if other_exp and other_exp != '0' else 'Not specified'}
Revenue Stage Expertise: {revenue_stage or 'Not specified'}
Growth Ceiling Story: {growth_story[:400] if growth_story else 'Not specified'}
Document Summary: {doc_summary}
---
"""

    prompt = f"""
You are helping an Indian founder find the right expert.

Founder's requirement:
"{query}"

Expert profiles to evaluate:
{expert_info}

TIER CLASSIFICATION RULES:

NEW DATA — USE FOR STRONGER MATCHING:
- "Core Business Problem They Can Advise On": Direct experience signal → strong Tier 1 indicator if aligned.
- "Active Industry Sectors": Use alongside Industry for industry matching.
- "Revenue Stage Expertise": Match to founder's stage.
- "Growth Ceiling Story": If ceiling matches founder's problem → very strong Tier 1 signal.

TIER 1 — Strong Match (min 1, max 5):
Both conditions MUST be true:
  ✅ Condition 1: Industry Match — worked IN the same or closely related industry.
  ✅ Condition 2: Operator Experience — PERSONALLY solved the specific problem the founder faces.
If even ONE condition is missing → Tier 2.

TIER 2 — Partial Match (max 5):
Meets at least ONE of:
  - Industry match but lacks operator experience in the specific problem
  - Operator experience in problem area but different industry
  - Strong relevant expertise that could still help

MINIMUM RESULT GUARANTEE:
- Return at least 1 expert total across both tiers. NEVER return an empty array.
- If no Tier 1, place best available in Tier 2 with honest Match Reason.

SCORING:
- Industry Match: 3 pts | Operator Experience: 3 pts | Relevant Expertise: 2 pts | Key Credentials: 2 pts

STRICT RULES:
- Never Tier 1 just because the expert is impressive
- Industry alone ≠ Tier 1 | Operator experience alone ≠ Tier 1 | BOTH required
- When in doubt → Tier 2

Return ONLY a JSON array (Tier 1 first, then Tier 2, max 10 total):
[
  {{
    "Tier": "1",
    "Name": "exact name",
    "Overall Score": "number only e.g. 8",
    "Industry Match Score": "score | one line e.g. 3 | Worked in manufacturing exports",
    "Hands On Score": "score | one line e.g. 3 | Personally handled DGFT documentation",
    "Expertise Score": "score | one line e.g. 2 | Strong supply chain expertise",
    "Credibility Score": "score | one line e.g. 2 | CII speaker, board member",
    "Core Expertise": "1 line — single most relevant core area",
    "Match Reason": "2-3 lines — why Tier 1 or 2 based on industry + operator experience",
    "Relevant Experience": "specific experience relevant to founder's problem",
    "Current Designation": "designation",
    "Current Organization": "organization",
    "Qualification": "qualification",
    "Hands On Experience": "Yes / No / Partial",
    "Hands On Details": "1-2 lines on what they personally did as an operator",
    "Core Problem Match": "1 line on alignment with founder's need",
    "Revenue Stage Fit": "1 line on revenue stage match",
    "Growth Ceiling Relevance": "1 line on growth ceiling story relevance"
  }}
]

Return only the JSON array. No extra text.
"""

    ai_raw  = call_ai(prompt, max_tokens=3000)
    cleaned = re.sub(r"```json|```", "", ai_raw).strip()
    try:
        results = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', cleaned, re.DOTALL)
        results = json.loads(m.group()) if m else []

    # Python-level fallback
    if not results:
        for _, row in candidates.head(2).iterrows():
            results.append({
                "Tier": "2", "Name": row.get("Name", "Unknown"),
                "Overall Score": "4",
                "Industry Match Score": "1 | Closest available — no strong industry alignment",
                "Hands On Score": "1 | Limited operator experience confirmed",
                "Expertise Score": "1 | Some relevant expertise may apply",
                "Credibility Score": "1 | Profile available for review",
                "Core Expertise": row.get("Primary Expertise", "Not specified"),
                "Match Reason": "No strong match found. Closest available by semantic similarity. Verify manually before outreach.",
                "Relevant Experience": row.get("Description", "")[:300],
                "Current Designation": row.get("Current Designation", ""),
                "Current Organization": row.get("Current Organization", ""),
                "Qualification": row.get("Qualification", ""),
                "Hands On Experience": "No",
                "Hands On Details": "Hands-on fit could not be confirmed. Please review full profile.",
                "Core Problem Match": "Not specified",
                "Revenue Stage Fit": "Not specified",
                "Growth Ceiling Relevance": "Not specified",
            })
    return results

# ─────────────────────────────────────────────
#  HELPERS – BROWSE TAB (Experience Type & Mentor Type)
# ─────────────────────────────────────────────
_GENERALIST_DOMAINS = [
    "operations", "finance", "legal", "hr", "human resources",
    "marketing", "sales", "product management", "product", "strategy",
    "supply chain", "accounting", "tax", "compliance", "branding",
    "customer success", "business development", "technology", "it",
    "people", "talent", "procurement", "general management"
]

def build_profile_text(row):
    parts = [
        row.get("Primary Expertise", ""), row.get("Secondary Expertise", ""),
        row.get("Description", ""),
        row.get("What is the one business problem you are most qualified to advise on from direct experience?", ""),
        row.get("Other Experience(s), if any", ""),
        row.get("Doc Text", ""), row.get("Current Designation", ""),
        row.get("Qualification", ""),
    ]
    return " | ".join(p for p in parts if p and str(p).strip() not in ("", "nan"))

def classify_experience_type_rule(profile_text):
    text = profile_text.lower()
    types = []
    if any(s in text for s in ["consult", "advisory", "advisor", "advise",
                                "mentor", "mentoring", "1:1", "coaching",
                                "strategic advice", "board advisor", "management consulting"]):
        types.append("Consulting/Advisory")
    if any(s in text for s in ["workshop", "masterclass", "master class", "webinar",
                                "1-to-many", "cohort", "bootcamp", "training program",
                                "seminar", "guest lecture", "keynote", "speaker",
                                "facilitator", "group session", "panel"]):
        types.append("Workshops/Masterclass/1-M Events")
    return types if types else ["Consulting/Advisory"]

def classify_sme_generalist_rule(row):
    combined = (
        row.get("Primary Expertise", "") + " " + row.get("Secondary Expertise", "") + " " +
        row.get("Description", "")
    ).lower()
    matched = {d for d in _GENERALIST_DOMAINS if d in combined}
    return "Generalist" if len(matched) >= 4 else "SME"

@st.cache_data(show_spinner=False)
def classify_all_mentors_ai(df_hash: str) -> dict:
    """AI-based batch classification. Cached per data hash."""
    names_and_profiles = []
    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if name and name.lower() not in ("nan", ""):
            names_and_profiles.append((name, build_profile_text(row)))

    results = {}
    batch_size = 20
    for start in range(0, len(names_and_profiles), batch_size):
        batch = names_and_profiles[start: start + batch_size]
        profiles_json = json.dumps(
            [{"name": n, "profile": p[:600]} for n, p in batch],
            ensure_ascii=False
        )
        prompt = f"""Classify each mentor profile for a startup accelerator.

For each mentor determine:
1. experience_types (array, one or both):
   - "Consulting/Advisory": 1:1 advisory, consulting, coaching, mentoring
   - "Workshops/Masterclass/1-M Events": workshops, masterclasses, webinars, bootcamps, group sessions, speaking

2. mentor_type (exactly one):
   - "SME": deep expertise in 1-3 specific domains
   - "Generalist": broad exposure across 4+ distinct business functions

Defaults if sparse: experience_types=["Consulting/Advisory"], mentor_type="SME"

Respond ONLY with a valid JSON array, no markdown:
[{{"name":"...","experience_types":["..."],"mentor_type":"..."}}]

Profiles:
{profiles_json}"""
        try:
            raw    = call_ai(prompt, max_tokens=2000)
            raw    = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            for item in parsed:
                results[item["name"]] = {
                    "experience_types": item.get("experience_types", ["Consulting/Advisory"]),
                    "mentor_type":      item.get("mentor_type", "SME")
                }
        except Exception:
            for name, profile in batch:
                results[name] = {
                    "experience_types": classify_experience_type_rule(profile),
                    "mentor_type": "SME"
                }
    return results

@st.cache_data(show_spinner=False)
def build_directory(df_hash: str) -> pd.DataFrame:
    classifications = classify_all_mentors_ai(df_hash)
    rows = []
    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        cls = classifications.get(name, {
            "experience_types": classify_experience_type_rule(build_profile_text(row)),
            "mentor_type":      classify_sme_generalist_rule(row)
        })
        linkedin_raw = str(row.get("LinkedIn", "")).strip()
        linkedin_display = linkedin_raw if linkedin_raw and linkedin_raw.lower() not in ("nan", "none", "") else ""

        exp_parts = [str(row.get("Primary Expertise", "")).strip(), str(row.get("Secondary Expertise", "")).strip()]
        expertise_str = " · ".join(p for p in exp_parts if p and p.lower() not in ("nan", ""))

        ind_parts = [str(row.get("Primary Industry", "")).strip(), str(row.get("Secondary Industry", "")).strip()]
        industry_str = " · ".join(p for p in ind_parts if p and p.lower() not in ("nan", ""))

        yoe_raw = str(row.get("Years of Experience", "")).strip()
        try:
            yoe = int(float(yoe_raw)) if yoe_raw.lower() not in ("nan", "none", "") else None
        except (ValueError, TypeError):
            yoe = None

        prog = str(row.get("Program", "")).strip()
        prog = prog if prog.lower() not in ("nan", "none", "") else "—"

        rows.append({
            "Name":                name,
            "LinkedIn":            linkedin_display,
            "Primary Expertise":   expertise_str or "—",
            "Primary Industry":    industry_str or "—",
            "Years of Experience": yoe,
            "Program":             prog,
            "Experience Type":     ", ".join(cls["experience_types"]),
            "Mentor Type":         cls["mentor_type"],
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_search, tab_browse = st.tabs(["🔍 AI Mentor Search", "📋 Browse & Filter"])

# ════════════════════════════════════════════
#  TAB 1 — AI SEARCH (Tier 1 / Tier 2)
# ════════════════════════════════════════════
with tab_search:
    st.markdown(
        "Describe your **founder's requirement** — industry, problem, stage — "
        "and I'll recommend the best-fit mentors as **Tier 1** (strong match) "
        "and **Tier 2** (partial match)."
    )
    st.markdown("---")

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "recommendations":
                st.markdown(message["summary"])
                display_expert_results(message["recommendations"], df)
            elif message.get("type") == "retry_prompt":
                st.markdown(message["content"])
                if message == st.session_state.messages[-1] and st.session_state.pending_retry:
                    cy, cn, _ = st.columns([1, 1, 4])
                    with cy:
                        if st.button("✅ Yes, retry", key="hist_retry_yes"):
                            rq = st.session_state.retry_query
                            st.session_state.pending_retry = False
                            st.session_state.messages.append({"role": "user", "content": "Yes"})
                            with st.spinner("Retrying..."):
                                try:
                                    rr = run_search(rq, filtered_df, filtered_vectors)
                                    st.session_state.last_recommendations = rr
                                    st.session_state.last_query = rq
                                    t1 = len([r for r in rr if r.get("Tier") == "1"])
                                    t2 = len([r for r in rr if r.get("Tier") == "2"])
                                    save_to_history(rq, t1, t2)
                                    st.session_state.messages.append({
                                        "role": "assistant", "type": "recommendations",
                                        "summary": f"Found **{t1} Tier 1** and **{t2} Tier 2** expert(s).",
                                        "recommendations": rr,
                                        "content": f"Found **{t1} Tier 1** and **{t2} Tier 2** expert(s)."
                                    })
                                except Exception as e:
                                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"Error: {e}"})
                            st.rerun()
                    with cn:
                        if st.button("❌ No, cancel", key="hist_retry_no"):
                            st.session_state.pending_retry = False
                            st.session_state.retry_query = ""
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": "No problem! Try a new search anytime."})
                            st.rerun()
            else:
                st.markdown(message["content"])

    # Re-run from sidebar history
    if st.session_state.get("_rerun_query"):
        user_input = st.session_state._rerun_query
        st.session_state._rerun_query = None
    else:
        user_input = None

    chat_input = st.chat_input("Describe the requirement or ask a follow-up…")
    if chat_input:
        user_input = chat_input

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        intent = detect_intent(user_input, st.session_state.last_recommendations)

        # ── Follow-up ──
        if intent == "followup":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    conv_history = ""
                    for msg in st.session_state.messages[-6:]:
                        role    = "Founder" if msg["role"] == "user" else "Assistant"
                        content = msg.get("content", msg.get("summary", ""))
                        conv_history += f"{role}: {content}\n"

                    fp = f"""You are an AI expert-matching assistant for Resources Network.

Original search: "{st.session_state.last_query}"
Conversation so far:\n{conv_history}
Current recommendations:\n{json.dumps(st.session_state.last_recommendations, indent=2)}
Follow-up: "{user_input}"

Instructions:
- Answer conversationally and helpfully
- Always mention Tier when referencing an expert
- Compare experts clearly if asked (pros/cons, industry match, operator experience)
- Reference names, designations, scores where relevant
- Do NOT return JSON — return a natural response"""
                    try:
                        resp = call_ai(fp, max_tokens=1024)
                        st.markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": resp})
                    except Exception as e:
                        st.error(f"AI Error: {e}")

        # ── New search ──
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching for the best mentors…"):
                    try:
                        recs = run_search(user_input, filtered_df, filtered_vectors)
                        if not recs:
                            st.session_state.pending_retry = True
                            st.session_state.retry_query   = user_input
                            retry_msg = (
                                "🔍 Couldn't retrieve results — encountered a glitch. "
                                "Shall I try again?\n\n**Type Yes or No, or use the buttons below.**"
                            )
                            st.markdown(retry_msg)
                            ry, rn, _ = st.columns([1, 1, 4])
                            with ry:
                                if st.button("✅ Yes, retry", key="inline_retry_yes"):
                                    st.session_state.pending_retry = False
                                    st.session_state.messages.append({"role": "user", "content": "Yes"})
                                    with st.spinner("Retrying..."):
                                        try:
                                            rr = run_search(user_input, filtered_df, filtered_vectors)
                                            st.session_state.last_recommendations = rr
                                            st.session_state.last_query = user_input
                                            t1 = len([r for r in rr if r.get("Tier") == "1"])
                                            t2 = len([r for r in rr if r.get("Tier") == "2"])
                                            save_to_history(user_input, t1, t2)
                                            st.session_state.messages.append({
                                                "role": "assistant", "type": "recommendations",
                                                "summary": f"Found **{t1} Tier 1** and **{t2} Tier 2** expert(s).",
                                                "recommendations": rr,
                                                "content": f"Found **{t1} Tier 1** and **{t2} Tier 2** expert(s)."
                                            })
                                        except Exception as e:
                                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"Error: {e}"})
                                    st.rerun()
                            with rn:
                                if st.button("❌ No, cancel", key="inline_retry_no"):
                                    st.session_state.pending_retry = False
                                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": "No problem!"})
                                    st.rerun()
                            st.session_state.messages.append({"role": "assistant", "type": "retry_prompt", "content": retry_msg})

                        else:
                            st.session_state.last_recommendations = recs
                            st.session_state.last_query           = user_input
                            st.session_state.pending_retry        = False
                            t1 = len([m for m in recs if m.get("Tier") == "1"])
                            t2 = len([m for m in recs if m.get("Tier") == "2"])
                            save_to_history(user_input, t1, t2)
                            prog_note = f" *(filtered to: {', '.join(selected_programs)})*" if selected_programs else ""
                            summary = (
                                f"Found **{t1} Tier 1 expert(s)** (Industry + Operator experience) and "
                                f"**{t2} Tier 2 expert(s)** (partial match){prog_note}.\n\n"
                                "You can **compare experts**, **ask about a specific mentor**, "
                                "**refine the search**, or **start a new search** anytime."
                            )
                            st.session_state.messages.append({
                                "role": "assistant", "type": "recommendations",
                                "summary": summary, "recommendations": recs, "content": summary
                            })
                            st.rerun()

                    except Exception as e:
                        st.session_state.pending_retry = True
                        st.session_state.retry_query   = user_input
                        retry_msg = f"🔍 Encountered an error: `{e}`. Shall I try again?"
                        st.markdown(retry_msg)
                        st.session_state.messages.append({"role": "assistant", "type": "retry_prompt", "content": retry_msg})

# ════════════════════════════════════════════
#  TAB 2 — BROWSE & FILTER
# ════════════════════════════════════════════
with tab_browse:
    st.markdown("Browse the full mentor pool. Filters are independent of the AI Search tab.")
    st.markdown("---")

    with st.spinner("🔄 Classifying mentor profiles… (first load only)"):
        df_hash = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()
        directory_df = build_directory(df_hash)

    if directory_df.empty:
        st.error("No mentor data found. Check that mentors.xlsx is uploaded correctly.")
        st.stop()

    st.success(f"✅ {len(directory_df)} mentors loaded.")

    # ── Filters (inline, not sidebar, so they don't conflict) ──
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        all_names = sorted(directory_df["Name"].dropna().unique().tolist())
        sel_names = st.multiselect("👤 Mentor Name", options=all_names,
                                   placeholder="All mentors", key="browse_names")
    with fc2:
        browse_programs = sorted(set(
            p.strip()
            for val in directory_df["Program"]
            for p in str(val).split(",")
            if p.strip() and p.strip() not in ("—", "nan", "none", "")
        ))
        sel_progs = st.multiselect("📌 Program", options=browse_programs,
                                   placeholder="All programs", key="browse_progs")
    with fc3:
        sel_exp_types = st.multiselect(
            "🛠️ Experience Type",
            options=["Consulting/Advisory", "Workshops/Masterclass/1-M Events"],
            placeholder="All types", key="browse_exp"
        )
    with fc4:
        sel_mentor_types = st.multiselect(
            "🎯 Mentor Type",
            options=["SME", "Generalist"],
            placeholder="SME & Generalist", key="browse_mtype"
        )

    # ── Apply filters ──
    filt = directory_df.copy()
    if sel_names:
        filt = filt[filt["Name"].isin(sel_names)]
    if sel_progs:
        filt = filt[filt["Program"].apply(
            lambda x: any(s in [p.strip() for p in str(x).split(",")] for s in sel_progs)
        )]
    if sel_exp_types:
        filt = filt[filt["Experience Type"].apply(
            lambda x: any(s in str(x) for s in sel_exp_types)
        )]
    if sel_mentor_types:
        filt = filt[filt["Mentor Type"].isin(sel_mentor_types)]

    st.markdown(f"**Showing {len(filt)} mentor(s)**")

    if filt.empty:
        st.info("No mentors match the selected filters. Try removing some filters.")
    else:
        display_cols = ["Name", "LinkedIn", "Primary Expertise", "Primary Industry",
                        "Years of Experience", "Program", "Experience Type", "Mentor Type"]

        def style_mentor_type(val):
            if val == "Generalist":
                return "background-color:#E8F4FD;color:#1A5276;font-weight:600;"
            elif val == "SME":
                return "background-color:#EAFAF1;color:#1E8449;font-weight:600;"
            return ""

        def style_exp_type(val):
            if "Workshops" in str(val) and "Consulting" in str(val):
                return "background-color:#FEF9E7;color:#7D6608;font-weight:600;"
            elif "Workshops" in str(val):
                return "background-color:#FDF2F8;color:#7D3C98;font-weight:600;"
            elif "Consulting" in str(val):
                return "background-color:#EAF2FF;color:#154360;font-weight:600;"
            return ""

        try:
            styled = (
                filt[display_cols].style
                .map(style_mentor_type, subset=["Mentor Type"])
                .map(style_exp_type,    subset=["Experience Type"])
            )
        except AttributeError:
            styled = (
                filt[display_cols].style
                .applymap(style_mentor_type, subset=["Mentor Type"])
                .applymap(style_exp_type,    subset=["Experience Type"])
            )

        st.dataframe(
            styled,
            use_container_width=True,
            height=min(60 + len(filt) * 38, 700),
            column_config={
                "LinkedIn": st.column_config.LinkColumn("LinkedIn", display_text="🔗 Profile"),
                "Years of Experience": st.column_config.NumberColumn("Yrs of Exp"),
            },
            hide_index=True,
        )

        # Download
        st.markdown("---")
        export_df = filt[display_cols].copy()
        st.download_button(
            label="⬇️ Download filtered list as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="mentor_directory_filtered.csv",
            mime="text/csv"
        )

    with st.expander("ℹ️ How are Experience Type and Mentor Type determined?"):
        st.markdown("""
**Experience Type** — AI-classified from each mentor's profile text:
- **Consulting/Advisory**: 1:1 advisory, coaching, consulting, mentoring
- **Workshops/Masterclass/1-M Events**: group formats — workshops, masterclasses, webinars, bootcamps, speaking
- A mentor can have **both** types.

**Mentor Type** — AI-classified from breadth of domain coverage:
- **SME**: Deep specialist in 1–3 specific domains
- **Generalist**: Broad across 4+ business functions (Operations, Finance, Marketing, Sales, HR, Legal, Product, Strategy…)

> ⚠️ Classifications are AI-generated from available profile text. Quality depends on how rich the Excel data is.
        """)
