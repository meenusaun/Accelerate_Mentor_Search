import streamlit as st
import pandas as pd
import anthropic
import os
import json
import re
from io import BytesIO
import pdfplumber
import docx

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Mentor Directory - Browse & Filter",
    page_icon="📋",
    layout="wide"
)

# ------------------ PASSWORD PROTECTION ------------------
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

# ------------------ CLIENTS ------------------
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ------------------ DOCUMENT EXTRACTION UTILS ------------------
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""
    return text.strip()

def extract_text_from_docx_bytes(file_bytes):
    text = ""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception:
        text = ""
    return text.strip()

def extract_text_from_file_path(file_path):
    if not file_path or not isinstance(file_path, str) or file_path.strip() == "":
        return ""
    file_path = file_path.strip()
    try:
        if file_path.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                return extract_text_from_pdf_bytes(f.read())
        elif file_path.lower().endswith(".docx"):
            with open(file_path, "rb") as f:
                return extract_text_from_docx_bytes(f.read())
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""
    return ""

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_excel(
        "mentors.xlsx",
        engine="openpyxl",
        dtype=str,
        na_filter=False
    )

    required_cols = [
        "Name", "LinkedIn", "Expertise", "Secondary Expertise",
        "Industry", "Secondary Industry", "Description",
        "Expertise Tags", "Industry Tags", "Document Path",
        "Program", "Years of Experience", "Location",
        "Current Organization", "Current Designation", "Qualification",
        "What is the one business problem you are most qualified to advise on from direct experience?",
        "Other Experience(s), if any",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    is_cloud = (
        os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"
        or os.environ.get("IS_STREAMLIT_CLOUD") is not None
        or not os.path.exists(os.path.expanduser("~/.streamlit/config.toml"))
    )
    if is_cloud:
        df["Doc Text"] = ""
    else:
        df["Doc Text"] = df["Document Path"].apply(extract_text_from_file_path)

    return df

df = load_data()

# ------------------ CLASSIFICATION ENGINE ------------------
_GENERALIST_DOMAINS = [
    "operations", "finance", "legal", "hr", "human resources",
    "marketing", "sales", "product management", "product", "strategy",
    "supply chain", "accounting", "tax", "compliance", "branding",
    "customer success", "business development", "technology", "it",
    "people", "talent", "procurement", "admin", "general management"
]

def build_profile_text(row):
    """Combine all relevant text fields for classification."""
    parts = [
        row.get("Expertise", ""),
        row.get("Secondary Expertise", ""),
        row.get("Description", ""),
        row.get("Expertise Tags", ""),
        row.get("What is the one business problem you are most qualified to advise on from direct experience?", ""),
        row.get("Other Experience(s), if any", ""),
        row.get("Doc Text", ""),
        row.get("Current Designation", ""),
        row.get("Qualification", ""),
    ]
    return " | ".join(p for p in parts if p and str(p).strip() not in ("", "nan"))

def classify_experience_type_rule(profile_text: str) -> list[str]:
    """
    Rule-based classification for experience type.
    Returns a list: one or both of 'Consulting/Advisory', 'Workshops/Masterclass/1-M Events'.
    """
    text = profile_text.lower()
    types = []

    consulting_signals = [
        "consult", "advisory", "advisor", "advise", "counsel",
        "mentor", "mentoring", "1:1", "one-on-one", "coaching",
        "strategic advice", "business advice", "independent advisor",
        "board advisor", "management consulting"
    ]
    workshop_signals = [
        "workshop", "masterclass", "master class", "webinar",
        "1-to-many", "1 to many", "cohort", "bootcamp", "boot camp",
        "training program", "seminar", "guest lecture", "keynote",
        "speaker", "facilitator", "facilitating", "group session",
        "community program", "panel", "conference talk"
    ]

    if any(s in text for s in consulting_signals):
        types.append("Consulting/Advisory")
    if any(s in text for s in workshop_signals):
        types.append("Workshops/Masterclass/1-M Events")

    # Default fallback: if nothing detected, mark as Consulting/Advisory
    # (most mentors in a curated pool are advisory by default)
    if not types:
        types.append("Consulting/Advisory")

    return types

def classify_sme_or_generalist_rule(row) -> str:
    """
    Rule-based classification: SME vs Generalist.
    Generalist = profile text covers 4+ distinct business function domains.
    SME = specialises in 1–3 domains.
    """
    combined = (
        row.get("Expertise", "") + " " +
        row.get("Secondary Expertise", "") + " " +
        row.get("Description", "") + " " +
        row.get("Expertise Tags", "")
    ).lower()

    matched_domains = set()
    for domain in _GENERALIST_DOMAINS:
        if domain in combined:
            matched_domains.add(domain)

    return "Generalist" if len(matched_domains) >= 4 else "SME"

@st.cache_data(show_spinner=False)
def classify_all_mentors_ai(names_and_profiles: list[tuple]) -> dict:
    """
    AI-based batch classification. Returns dict: name -> {experience_types, mentor_type}.
    Processes in batches of 20 to keep prompts manageable.
    Falls back to rule-based if AI call fails.
    """
    results = {}
    batch_size = 20

    for batch_start in range(0, len(names_and_profiles), batch_size):
        batch = names_and_profiles[batch_start: batch_start + batch_size]

        profiles_json = json.dumps([
            {"name": name, "profile": profile[:600]}
            for name, profile in batch
        ], ensure_ascii=False)

        prompt = f"""You are classifying mentor profiles for a startup accelerator program.

For each mentor below, determine:

1. **experience_types** — which of these apply (can be one OR both):
   - "Consulting/Advisory": 1:1 advisory, consulting, coaching, mentoring sessions
   - "Workshops/Masterclass/1-M Events": workshops, masterclasses, webinars, bootcamps, group sessions, speaking/facilitation

2. **mentor_type** — pick ONE:
   - "SME" (Subject Matter Expert): deep expertise in 1-3 specific domains
   - "Generalist": broad exposure across 4+ distinct business functions (e.g. operations, finance, marketing, HR, sales, product, legal, strategy, technology)

Rules:
- If profile text is sparse or unclear, default experience_type to ["Consulting/Advisory"] and mentor_type to "SME"
- experience_types must be a JSON array with at least one value
- mentor_type must be exactly "SME" or "Generalist"

Respond ONLY with a valid JSON array. No markdown, no preamble. Format:
[
  {{"name": "...", "experience_types": ["..."], "mentor_type": "..."}},
  ...
]

Profiles:
{profiles_json}
"""
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            for item in parsed:
                results[item["name"]] = {
                    "experience_types": item.get("experience_types", ["Consulting/Advisory"]),
                    "mentor_type": item.get("mentor_type", "SME")
                }
        except Exception:
            # Fall back to rule-based for this batch
            for name, profile in batch:
                row_data = {}  # minimal fallback
                results[name] = {
                    "experience_types": classify_experience_type_rule(profile),
                    "mentor_type": "SME"
                }

    return results

# ------------------ BUILD DIRECTORY TABLE ------------------
@st.cache_data(show_spinner=False)
def build_directory(df_hash: str) -> pd.DataFrame:
    """Build the enriched mentor directory with classifications."""
    rows = []
    names_and_profiles = []

    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        profile_text = build_profile_text(row)
        names_and_profiles.append((name, profile_text))

    # AI classification
    classifications = classify_all_mentors_ai(names_and_profiles)

    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name.lower() in ("nan", ""):
            continue

        cls = classifications.get(name, {
            "experience_types": classify_experience_type_rule(build_profile_text(row)),
            "mentor_type": classify_sme_or_generalist_rule(row)
        })

        # LinkedIn: clean up
        linkedin_raw = str(row.get("LinkedIn", "")).strip()
        if linkedin_raw and linkedin_raw.lower() not in ("nan", "none", ""):
            linkedin_display = f"[Profile]({linkedin_raw})"
        else:
            linkedin_display = "—"

        # Expertise: primary + secondary
        exp_parts = [
            str(row.get("Expertise", "")).strip(),
            str(row.get("Secondary Expertise", "")).strip()
        ]
        expertise_str = " · ".join(p for p in exp_parts if p and p.lower() not in ("nan", ""))

        # Industry
        ind_parts = [
            str(row.get("Industry", "")).strip(),
            str(row.get("Secondary Industry", "")).strip()
        ]
        industry_str = " · ".join(p for p in ind_parts if p and p.lower() not in ("nan", ""))

        # Years of experience
        yoe = str(row.get("Years of Experience", "")).strip()
        if yoe.lower() in ("nan", "none", ""):
            yoe = "—"

        # Program
        prog = str(row.get("Program", "")).strip()
        if prog.lower() in ("nan", "none", ""):
            prog = "—"

        rows.append({
            "Name": name,
            "LinkedIn": linkedin_display,
            "Expertise": expertise_str or "—",
            "Industry": industry_str or "—",
            "Years of Experience": yoe,
            "Program": prog,
            "Experience Type": ", ".join(cls["experience_types"]),
            "Mentor Type": cls["mentor_type"],
            # Raw LinkedIn for link display
            "_linkedin_raw": linkedin_raw,
        })

    return pd.DataFrame(rows)

# ---- Header ----
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    try:
        st.image("DP_BG1.png", width=150)
    except Exception:
        pass
with col2:
    st.markdown(
        "<h2 style='text-align:center;'>📋 Mentor Directory — Browse & Filter</h2>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ---- Load/classify (with spinner) ----
with st.spinner("🔄 Loading mentor profiles and classifying experience types... (first load may take ~30 sec)"):
    import hashlib
    df_hash = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()
    directory_df = build_directory(df_hash)

if directory_df.empty:
    st.error("No mentor data found. Please check that mentors.xlsx is correctly uploaded.")
    st.stop()

st.success(f"✅ {len(directory_df)} mentors loaded.")

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("Use the filters below to narrow the mentor list.")

# -- Filter: Mentor Name --
all_names = sorted(directory_df["Name"].dropna().unique().tolist())
selected_names = st.sidebar.multiselect(
    "👤 Mentor Name",
    options=all_names,
    default=[],
    placeholder="All mentors"
)

# -- Filter: Program --
all_programs = sorted(set(
    p.strip()
    for val in directory_df["Program"]
    for p in str(val).split(",")
    if p.strip() and p.strip() not in ("—", "nan", "none", "")
))
selected_programs = st.sidebar.multiselect(
    "📌 Program",
    options=all_programs,
    default=[],
    placeholder="All programs"
)

# -- Filter: Experience Type --
exp_type_options = ["Consulting/Advisory", "Workshops/Masterclass/1-M Events"]
selected_exp_types = st.sidebar.multiselect(
    "🛠️ Experience Type",
    options=exp_type_options,
    default=[],
    placeholder="All experience types"
)

# -- Filter: Mentor Type --
mentor_type_options = ["SME", "Generalist"]
selected_mentor_types = st.sidebar.multiselect(
    "🎯 Mentor Type",
    options=mentor_type_options,
    default=[],
    placeholder="SME & Generalist"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Experience Type** is AI-classified from mentor profiles.\n\n"
    "**SME** = Subject Matter Expert (deep expertise in 1–3 domains)\n\n"
    "**Generalist** = Broad exposure across 4+ business functions"
)

# ------------------ APPLY FILTERS ------------------
filtered = directory_df.copy()

if selected_names:
    filtered = filtered[filtered["Name"].isin(selected_names)]

if selected_programs:
    def prog_match(prog_val):
        mentor_progs = [p.strip() for p in str(prog_val).split(",")]
        return any(sel in mentor_progs for sel in selected_programs)
    filtered = filtered[filtered["Program"].apply(prog_match)]

if selected_exp_types:
    def exp_match(exp_val):
        return any(sel in str(exp_val) for sel in selected_exp_types)
    filtered = filtered[filtered["Experience Type"].apply(exp_match)]

if selected_mentor_types:
    filtered = filtered[filtered["Mentor Type"].isin(selected_mentor_types)]

# ------------------ DISPLAY TABLE ------------------
st.markdown(f"### Showing **{len(filtered)}** mentor(s)")

if filtered.empty:
    st.info("No mentors match the selected filters. Try removing some filters.")
else:
    # Display columns (hide _linkedin_raw)
    display_cols = ["Name", "LinkedIn", "Expertise", "Industry",
                    "Years of Experience", "Program", "Experience Type", "Mentor Type"]

    # Color-code Mentor Type
    def style_mentor_type(val):
        if val == "Generalist":
            return "background-color: #E8F4FD; color: #1A5276; font-weight: 600;"
        elif val == "SME":
            return "background-color: #EAFAF1; color: #1E8449; font-weight: 600;"
        return ""

    def style_exp_type(val):
        if "Workshop" in str(val) and "Consulting" in str(val):
            return "background-color: #FEF9E7; color: #7D6608; font-weight: 600;"
        elif "Workshop" in str(val):
            return "background-color: #FDF2F8; color: #7D3C98; font-weight: 600;"
        elif "Consulting" in str(val):
            return "background-color: #EAF2FF; color: #154360; font-weight: 600;"
        return ""

    try:
        # pandas >= 2.1 uses .map(); older versions use .applymap()
        styled = (
            filtered[display_cols]
            .style
            .map(style_mentor_type, subset=["Mentor Type"])
            .map(style_exp_type, subset=["Experience Type"])
        )
    except AttributeError:
        styled = (
            filtered[display_cols]
            .style
            .applymap(style_mentor_type, subset=["Mentor Type"])
            .applymap(style_exp_type, subset=["Experience Type"])
        )

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(60 + len(filtered) * 38, 700),
        column_config={
            "LinkedIn": st.column_config.LinkColumn(
                "LinkedIn",
                display_text="🔗 Profile"
            ),
        },
        hide_index=True,
    )

    # ------------------ DOWNLOAD BUTTON ------------------
    st.markdown("---")
    export_df = filtered[display_cols].copy()
    # Replace markdown links with raw URLs for export
    export_df["LinkedIn"] = filtered["_linkedin_raw"].values

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered list as CSV",
        data=csv_bytes,
        file_name="mentor_directory_filtered.csv",
        mime="text/csv"
    )

# ------------------ LEGEND ------------------
st.markdown("---")
with st.expander("ℹ️ How are Experience Type and Mentor Type determined?"):
    st.markdown("""
**Experience Type** is classified by AI from each mentor's profile text (expertise description, tags, bio):
- **Consulting/Advisory**: 1:1 advisory, coaching, consulting, mentoring sessions
- **Workshops/Masterclass/1-M Events**: group formats — workshops, masterclasses, webinars, bootcamps, speaking
- A mentor can have **both** types if their profile reflects both formats.

**Mentor Type** is classified by AI based on breadth of domain coverage:
- **SME (Subject Matter Expert)**: Deep specialist in 1–3 specific domains (e.g. Fundraising, D2C Marketing, IP Law)
- **Generalist**: Broad exposure across 4+ distinct business functions (Operations, Finance, Marketing, Sales, HR, Legal, Product, Strategy, etc.)

> ⚠️ These classifications are AI-generated from available profile text and may not be 100% accurate. 
> If a mentor's type looks incorrect, it may be because their profile text is sparse or uses non-standard terminology.
""")
