"""scoring.py - all ranking logic in one place (used by the Colab notebook AND app.py)."""
import re
from datetime import datetime

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# EDIT ME: add skills for your field (lowercase)
SKILLS = [
    # tech
    "python", "java", "sql", "javascript", "html", "css", "react", "aws", "docker", "linux",
    "git", "machine learning", "data analysis", "networking", "testing", "sap", "autocad",
    # office / business
    "excel", "microsoft office", "powerpoint", "outlook", "project management",
    "budgeting", "forecasting", "reporting", "negotiation", "presentations", "scheduling",
    # sales / marketing / pr
    "sales", "crm", "customer service", "business development", "marketing", "social media",
    "seo", "public relations", "event planning", "media relations", "advertising",
    # finance / accounting / banking
    "accounting", "bookkeeping", "quickbooks", "accounts payable", "accounts receivable",
    "payroll", "auditing", "financial analysis", "tax", "reconciliation", "banking", "loans",
    # hr
    "recruitment", "onboarding", "employee relations", "benefits administration",
    # teaching / health / fitness
    "lesson planning", "curriculum", "classroom management", "patient care", "cpr",
    "electronic medical records", "nutrition", "personal training", "fitness",
    # design / media / arts
    "photoshop", "illustrator", "indesign", "graphic design", "video editing", "typography",
    # engineering / construction / aviation / automobile / agriculture / food
    "safety", "quality control", "maintenance", "blueprints", "welding", "construction",
    "aircraft", "flight operations", "diagnostics", "inventory", "logistics", "supply chain",
    "food safety", "menu planning", "crop", "irrigation", "legal research", "litigation",
]

# EDIT ME: how much each part counts (rescaled to add up to 1.0)
WEIGHTS = {"similarity": 0.35, "skills": 0.35, "category": 0.10, "experience": 0.10, "education": 0.10}

EDU_PATTERNS = [(4, r"ph\.?d|doctorate"),
                (3, r"master|m\.?tech|msc|m\.sc|mba|\bm\.[as]\b"),
                (2, r"bachelor|b\.?tech|bsc|b\.sc|\bb\.e\b|bca|b\.com|\bb\.[as]\b"),
                (1, r"diploma|associates? degree|associate of")]


def clean_resume(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text):
    return {s for s in SKILLS if re.search(rf"(?<![a-z0-9+#]){re.escape(s)}(?![a-z0-9+#])", text)}


def extract_experience(text):
    """Years of experience from '5 years', '18 months', or job date ranges like '2010 to 2014'."""
    this_year = datetime.now().year
    years = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b", text)]
    months = [float(n) / 12 for n in re.findall(r"(\d+(?:\.\d+)?)\s*months?\b", text)]
    explicit = max(years + months, default=0.0)

    starts, ends = [], []
    pattern = r"\b((?:19|20)\d{2})\s*(?:to|-|until)\s*(?:\d{1,2}\s+)?((?:19|20)\d{2}|current|present|now)\b"
    for s, e in re.findall(pattern, text):
        s = int(s)
        e = this_year if e in ("current", "present", "now") else int(e)
        if 1970 <= s <= e <= this_year:
            starts.append(s)
            ends.append(e)
    span = (max(ends) - min(starts)) if starts else 0
    return float(min(max(explicit, span), 40.0))


def extract_education(text):
    for level, pattern in EDU_PATTERNS:
        if re.search(pattern, text):
            return level
    return 0


def rank_candidates(model, jd_text, resumes, weights=None):
    """
    model   : trained pipeline (TF-IDF + classifier)
    jd_text : job description (string)
    resumes : dict {candidate name: raw resume text}
    Returns (table sorted best to worst, predicted job category, skills found in the job description)
    """
    w = weights or WEIGHTS
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}

    names = list(resumes)
    cleaned = [clean_resume(t) for t in resumes.values()]
    jd_clean = clean_resume(jd_text)

    tfidf = model.named_steps["tfidf"]
    sim = cosine_similarity(tfidf.transform([jd_clean]), tfidf.transform(cleaned))[0]
    jd_cat = model.predict([jd_clean])[0]
    cats = model.predict(cleaned)

    need = extract_skills(jd_clean)
    req_exp, req_edu = extract_experience(jd_clean), extract_education(jd_clean)

    rows = []
    for i, text in enumerate(cleaned):
        have = extract_skills(text)
        exp, edu = extract_experience(text), extract_education(text)
        skill_score = len(need & have) / len(need) if need else 0.0
        exp_score = min(exp / req_exp, 1.0) if req_exp else 1.0
        edu_score = 1.0 if edu >= req_edu else 0.0
        score = 100 * (w["similarity"] * min(sim[i] * 2, 1.0)
                       + w["skills"] * skill_score
                       + w["category"] * int(cats[i] == jd_cat)
                       + w["experience"] * exp_score
                       + w["education"] * edu_score)
        rows.append({"candidate": names[i], "score": round(score, 1), "category": cats[i],
                     "skills_%": round(100 * skill_score), "experience_yrs": round(exp, 1),
                     "matched_skills": ", ".join(sorted(need & have)) or "-",
                     "missing_skills": ", ".join(sorted(need - have)) or "-"})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.index += 1
    out.index.name = "rank"
    return out, jd_cat, need
