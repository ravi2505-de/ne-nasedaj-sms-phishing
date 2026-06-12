"""
Ne Nasedaj — SMS Phishing Detector
Bilingual version (EN/SR) with language toggle
Ensemble: TF-IDF/LogReg + fine-tuned BERTić classifier
"""
import gradio as gr
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# =========================
# LOAD DATASETS & DOMAINS
# =========================
try:
    df = pd.read_csv("sms_dataset_lc.csv")
except FileNotFoundError:
    df = pd.DataFrame({"text": [], "label": []})

try:
    phishing_domains_df = pd.read_csv("phishing_domains.csv")
    typosquatting_domains_df = pd.read_csv("typosquatting_domains.csv")
    PHISHING_DOMAINS = set(phishing_domains_df['domain'].str.lower().tolist())
    TYPOSQUATTING_DOMAINS = set(typosquatting_domains_df['domain'].str.lower().tolist())
    ALL_SUSPICIOUS_DOMAINS = PHISHING_DOMAINS.union(TYPOSQUATTING_DOMAINS)
except Exception:
    ALL_SUSPICIOUS_DOMAINS = set()

# =========================
# TF-IDF / LOGREG MODEL (Model A)
# =========================
if len(df) > 0:
    tfidf_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            lowercase=True,
            max_features=10000
        )),
        ("clf", LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    tfidf_model.fit(df["text"], df["label"])
else:
    tfidf_model = None

# --- BERTić model integration ---
BERTIC_MODEL_ID = "ravi2505/ne-nasedaj-sms-phishing"
BERTIC_DEVICE = torch.device("cpu")

try:
    bertic_tokenizer = AutoTokenizer.from_pretrained(BERTIC_MODEL_ID)
    bertic_model = AutoModelForSequenceClassification.from_pretrained(BERTIC_MODEL_ID)
    bertic_model.to(BERTIC_DEVICE)
    bertic_model.eval()
    BERTIC_AVAILABLE = True
except Exception as e:
    print(f"BERTić model unavailable: {e}")
    bertic_tokenizer = None
    bertic_model = None
    BERTIC_AVAILABLE = False


def predict_with_bertic(text, language="sr"):
    """Return the BERTić phishing prediction for raw SMS text."""
    if not BERTIC_AVAILABLE:
        raise RuntimeError("BERTić model is not available")

    prepared_text = f"[{language}] {text}"
    inputs = bertic_tokenizer(
        prepared_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {key: value.to(BERTIC_DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        logits = bertic_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    legitimate_probability = probs[0].item()
    phishing_probability = probs[1].item()
    label = "phishing" if phishing_probability >= legitimate_probability else "legitimate"
    confidence = max(phishing_probability, legitimate_probability)

    return {
        "label": label,
        "confidence": confidence,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
    }
# --- BERTić model integration ---
# =========================
# TRANSLATION DICTIONARIES
# =========================
TRANSLATIONS = {
    "en": {
        "title": "# 🛡️ NE NASEDAJ — SMS PHISHING DETECTOR",
        "subtitle": "### Ensemble AI analysis for SR/HR/BIH/CG",
        "sms_input_label": "✉️ Enter SMS message",
        "sms_placeholder": "Paste SMS message here...",
        "human_check": "I am not a robot",
        "privacy_check": "I accept the terms of use",
        "terms_link": "📄 Read terms of use",
        "analyze_btn": "🔍 Analyze",
        "output_label": "🔎 Result",
        "quick_tests": "💡 Quick tests",
        "phishing_title": "🚨 Phishing examples",
        "legit_title": "✅ Legitimate examples",
        "batch_title": "🧪 Batch testing — upload CSV",
        "batch_desc": "Upload `smishi_test_cases.csv` (columns: `text`, `label`, optional `technique`)",
        "upload_csv": "📂 Upload CSV",
        "run_batch": "▶️ Run batch test",
        "batch_output": "📊 Batch results",
        "footer": "Ne Nasedaj — protect yourself from scams 🛡️",
        "lang_toggle": "🌐 Language / Jezik",
        "threat_vectors": {
            "IBAN_PHISHING": {"label": "💳 IBAN Phishing", "desc": "Message asks for payment to an IBAN account."},
            "URGENCY": {"label": "⏰ Urgency", "desc": "Message uses time pressure or threat."},
            "CREDENTIAL_HARVEST": {"label": "🔑 Credential Harvest", "desc": "Message asks for personal or login details."},
            "FAKE_DELIVERY": {"label": "📦 Fake Delivery", "desc": "Impersonates package or customs notification."},
            "VISHING": {"label": "📞 Vishing", "desc": "Calls to contact a fake institution by phone."},
            "PRIZE_SCAM": {"label": "🎁 Prize Scam", "desc": "Fake notification about a prize or win."},
            "FINANCIAL_BAIT": {"label": "💰 Financial Bait", "desc": "Promises refund or financial gain."},
            "TYPOSQUATTING": {"label": "🌐 Typosquatting domain", "desc": "Link leads to fake site imitating a real institution."},
        },
        "len_too_short": "⚠️ Too short",
        "len_suspicious": "⚠️ Suspiciously short",
        "len_std": "✅ Standard length",
        "len_too_long": "⚠️ Too long",
        "len_too_short_desc": "Message is too short to be a real notification.",
        "len_suspicious_desc": "Real notifications are usually longer.",
        "len_std_desc": "Length corresponds to standard SMS.",
        "len_too_long_desc": "Standard SMS is {max_len} characters.",
        "status_phishing_high": "🚨 PHISHING — Do not fall for it!",
        "status_models_disagree": "⚠️ MODELS DISAGREE — Be cautious",
        "status_not_sure": "⚠️ NOT SURE — Be cautious",
        "status_possible_fraud": "⚠️ POSSIBLE FRAUD — Verify sender",
        "status_legit": "✅ LEGITIMATE — Looks safe",
        "why_both_phishing": "Both models indicate phishing.",
        "why_model_detected": "Model detected phishing attack patterns.",
        "why_disagree_tfidf": "TF-IDF model says <strong>{pred_a}</strong>, BERTić model says <strong>{pred_b}</strong>.",
        "why_low_confidence": "Models are not confident enough to make a clear decision.",
        "why_suspicious_features": "Message has suspicious features, but confidence is not high.",
        "why_legit": "Both models agree the message does not contain typical phishing patterns.",
        "why_red_flags": "<strong>Suspicious words:</strong> {red_flags}",
        "why_typosquatting": "<strong>Typosquatting domain:</strong> {domains}",
        "why_note_red_flags": "<strong>⚠️ Note:</strong> there are suspicious words ({red_flags}).",
        "action_phishing_high": "Do not click on links. Report the message to your operator.",
        "action_disagree": "Check with the sender directly before clicking or replying.",
        "action_not_sure": "Check with the sender directly before clicking or replying.",
        "action_possible_fraud": "Contact the institution directly via their official number.",
        "action_legit": "Verify the sender before clicking.",
        "copy_btn": "📋 Copy message",
        "copy_success": "✅ Copied!",
        "copy_error": "❌",
        "report_cert_btn": "🚨 Report to {name}",
        "report_police_btn": "👮 Report to {name}",
        "warn_contact_btn": "⚠️ Warn contact",
        "what_to_do": "🚀 What to do:",
        "why_label": "🔍 Why?",
        "advice_label": "💡 Advice:",
        "detected_vectors": "🏷️ Detected attack vectors:",
        "ensemble_confidence": "📊 Ensemble confidence",
        "model_tfidf": "🤖 TF-IDF:",
        "model_bertic": "🧠 BERTić:",
        "model_unavailable": "unavailable",
        "length_label": "📏 Length: {length} chars",
        "batch_technique": "Technique",
        "batch_message": "Message",
        "batch_expected": "Expected",
        "batch_verdict": "Verdict",
        "batch_models": "A/B models",
        "batch_vectors": "Vectors",
        "batch_ok": "OK?",
        "batch_summary": "📊 Batch test results (ensemble)",
        "batch_total": "Total:",
        "batch_correct": "Correct:",
        "batch_wrong": "Wrong:",
        "batch_uncertain": "Uncertain:",
        "batch_accuracy": "Accuracy:",
        "batch_uncertain_label": "⚠️ Models disagree / Not sure",
        "batch_phishing_label": "🚨 Phishing",
        "batch_possible_label": "⚠️ Possible fraud",
        "batch_legit_label": "✅ Legitimate",
        "terms_title": "📄 Terms of Use — Ne Nasedaj",
        "terms_p1": "<strong>1. Purpose</strong><br>Ne Nasedaj is a free SMS phishing detection tool for users in Serbia, Croatia, Bosnia and Herzegovina, and Montenegro.",
        "terms_p2": "<strong>2. Data Processing</strong><br>The message text you enter is analyzed only locally within the application. Messages are not stored, logged, or sent to external servers.",
        "terms_p3": "<strong>3. Limitation of Liability</strong><br>Analysis results are for informational purposes only and do not constitute a guarantee. The tool may make mistakes — always use common sense and verify the sender directly.",
        "terms_p4": "<strong>4. Prohibition of Misuse</strong><br>Using this tool for attack testing, generating phishing content, or any other malicious activity is prohibited.",
        "terms_p5": "<strong>5. Contact</strong><br>For questions and bug reports, contact us via the Hugging Face repository.",
        "terms_accept": "I understand and accept ✓",
        "ex_phishing_1": "Posta Serbia: Your package awaits customs. Pay 450 RSD: https://rs-posta.net",
        "ex_phishing_2": "Raiffeisen: Card blocked. Verify: https://raiffeisen-sigurno.com",
        "ex_legit_1": "Yettel: Subscription renewed. Balance: 0.00 RSD.",
        "ex_legit_2": "AIK Banka: E-token code 847392. Valid 60s.",
        "warning_text": "WARNING: This SMS has been identified as a potential phishing attack. Do not click links or enter personal information.",
    },
    "sr": {
        "title": "# 🛡️ NE NASEDAJ — SMS PHISHING DETECTOR",
        "subtitle": "### Ensemble AI analiza za SR/HR/BIH/CG",
        "sms_input_label": "✉️ Ukucaj SMS poruku",
        "sms_placeholder": "Nalepite tekst SMS poruke ovde...",
        "human_check": "Nisam robot",
        "privacy_check": "Prihvatam uslove korišćenja",
        "terms_link": "📄 Pročitaj uslove korišćenja",
        "analyze_btn": "🔍 Analiziraj",
        "output_label": "🔎 Rezultat",
        "quick_tests": "💡 Brzi testovi",
        "phishing_title": "🚨 Phishing primeri",
        "legit_title": "✅ Legitimni primeri",
        "batch_title": "🧪 Batch testiranje — učitaj CSV",
        "batch_desc": "Upload `smishi_test_cases.csv` (kolone: `text`, `label`, opciono `technique`)",
        "upload_csv": "📂 Učitaj CSV",
        "run_batch": "▶️ Pokreni batch test",
        "batch_output": "📊 Batch rezultati",
        "footer": "Ne Nasedaj — zaštitite se od prevara 🛡️",
        "lang_toggle": "🌐 Jezik / Language",
        "threat_vectors": {
            "IBAN_PHISHING": {"label": "💳 IBAN Phishing", "desc": "Poruka traži uplatu na IBAN račun."},
            "URGENCY": {"label": "⏰ Urgentnost", "desc": "Poruka koristi pritisak vremena ili prijetnju."},
            "CREDENTIAL_HARVEST": {"label": "🔑 Krađa podataka", "desc": "Poruka traži unos ličnih ili pristupnih podataka."},
            "FAKE_DELIVERY": {"label": "📦 Lažna dostava", "desc": "Imitacija obaveštenja o paketu ili carini."},
            "VISHING": {"label": "📞 Vishing", "desc": "Poziva na telefonski kontakt sa lažnom institucijom."},
            "PRIZE_SCAM": {"label": "🎁 Nagradna prevara", "desc": "Lažno obaveštenje o nagradi ili dobitku."},
            "FINANCIAL_BAIT": {"label": "💰 Finansijski mamac", "desc": "Obećava povrat novca ili finansijsku korist."},
            "TYPOSQUATTING": {"label": "🌐 Typosquatting domen", "desc": "Link vodi na lažni sajt koji imitira pravu instituciju."},
        },
        "len_too_short": "⚠️ Prekratko",
        "len_suspicious": "⚠️ Sumnjivo kratko",
        "len_std": "✅ Standardna dužina",
        "len_too_long": "⚠️ Predugo",
        "len_too_short_desc": "Poruka je prekratka da bi bila realno obaveštenje.",
        "len_suspicious_desc": "Realna obaveštenja su obično duža.",
        "len_std_desc": "Dužina odgovara standardnom SMS-u.",
        "len_too_long_desc": "Standardni SMS je {max_len} karaktera.",
        "status_phishing_high": "🚨 PHISHING — Ne nasedajte!",
        "status_models_disagree": "⚠️ MODELI SE NE SLAŽU — Budite oprezni",
        "status_not_sure": "⚠️ NISAM SIGURAN — Budite oprezni",
        "status_possible_fraud": "⚠️ MOGUĆA PREVARA — Proverite pošiljaoca",
        "status_legit": "✅ LEGITIMNO — Izgleda bezbedno",
        "why_both_phishing": "Oba modela ukazuju na phishing.",
        "why_model_detected": "Model je detektovao obrasce phishing napada.",
        "why_disagree_tfidf": "TF-IDF model kaže <strong>{pred_a}</strong>, BERTić model kaže <strong>{pred_b}</strong>.",
        "why_low_confidence": "Modeli nisu dovoljno sigurni da donesu jasnu odluku.",
        "why_suspicious_features": "Poruka ima sumnjive karakteristike, ali pouzdanost nije visoka.",
        "why_legit": "Oba modela se slažu da poruka ne sadrži tipične phishing obrasce.",
        "why_red_flags": "<strong>Sumnjive reči:</strong> {red_flags}",
        "why_typosquatting": "<strong>Typosquatting domen:</strong> {domains}",
        "why_note_red_flags": "<strong>⚠️ Napomena:</strong> postoje sumnjive reči ({red_flags}).",
        "action_phishing_high": "Ne klikajte na linkove. Prijavite poruku operateru.",
        "action_disagree": "Proverite pošiljaoca direktno pre nego što kliknete ili odgovorite.",
        "action_not_sure": "Proverite pošiljaoca direktno pre nego što kliknete ili odgovorite.",
        "action_possible_fraud": "Kontaktirajte instituciju direktno putem zvaničnog broja.",
        "action_legit": "Proverite pošiljaoca pre klika.",
        "copy_btn": "📋 Kopiraj poruku",
        "copy_success": "✅ Kopirano!",
        "copy_error": "❌",
        "report_cert_btn": "🚨 Prijavi {name}",
        "report_police_btn": "👮 Prijavi {name}",
        "warn_contact_btn": "⚠️ Upozori kontakt",
        "what_to_do": "🚀 Šta da radite:",
        "why_label": "🔍 Zašto?",
        "advice_label": "💡 Savet:",
        "detected_vectors": "🏷️ Detektovani vektori napada:",
        "ensemble_confidence": "📊 Pouzdanost ansambla",
        "model_tfidf": "🤖 TF-IDF:",
        "model_bertic": "🧠 BERTić:",
        "model_unavailable": "nedostupan",
        "length_label": "📏 Dužina: {length} kar.",
        "batch_technique": "Tehnika",
        "batch_message": "Poruka",
        "batch_expected": "Očekivano",
        "batch_verdict": "Verdikt",
        "batch_models": "A/B modeli",
        "batch_vectors": "Vektori",
        "batch_ok": "OK?",
        "batch_summary": "📊 Rezultati batch testa (ensemble)",
        "batch_total": "Ukupno:",
        "batch_correct": "Tačno:",
        "batch_wrong": "Pogrešno:",
        "batch_uncertain": "Nesigurno:",
        "batch_accuracy": "Tačnost:",
        "batch_uncertain_label": "⚠️ Modeli se ne slažu / Nisam siguran",
        "batch_phishing_label": "🚨 Phishing",
        "batch_possible_label": "⚠️ Moguća prevara",
        "batch_legit_label": "✅ Legitimno",
        "terms_title": "📄 Uslovi korišćenja — Ne Nasedaj",
        "terms_p1": "<strong>1. Svrha alata</strong><br>Ne Nasedaj je besplatni alat za detekciju SMS phishing poruka namenjen korisnicima u Srbiji, Hrvatskoj, Bosni i Hercegovini i Crnoj Gori.",
        "terms_p2": "<strong>2. Obrada podataka</strong><br>Tekst poruke koji unesete analizira se isključivo lokalno unutar aplikacije. Poruke se ne čuvaju, ne beleže i ne šalju na eksterne servere.",
        "terms_p3": "<strong>3. Ograničenje odgovornosti</strong><br>Rezultati analize su informativnog karaktera i ne predstavljaju garanciju. Alat može pogrešiti — uvek koristite zdrav razum i proverite pošiljaoca direktno.",
        "terms_p4": "<strong>4. Zabrana zloupotrebe</strong><br>Zabranjeno je korišćenje ovog alata u svrhe testiranja napada, generisanja phishing sadržaja ili bilo koje druge zlonamerne aktivnosti.",
        "terms_p5": "<strong>5. Kontakt</strong><br>Za pitanja i prijavu grešaka kontaktirajte nas putem Hugging Face repozitorijuma.",
        "terms_accept": "Razumem i prihvatam ✓",
        "ex_phishing_1": "Pošta Srbije: Vaš paket čeka na carini. Platite carinu od 450 RSD: https://rs-posta.net",
        "ex_phishing_2": "Raiffeisen: Kartica blokirana. Potvrdite: https://raiffeisen-sigurno.com",
        "ex_legit_1": "Yettel: Pretplata obnovljena. Stanje: 0,00 RSD.",
        "ex_legit_2": "AIK Banka: E-token kod 847392. Važi 60s.",
        "warning_text": "UPOZORENJE: Ova SMS poruka je identifikovana kao potencijalni phishing napad. Ne klikajte na linkove i ne unosite lične podatke.",
    }
}

# =========================
# LOCALIZED THREAT VECTORS
# =========================
def get_localized_threat_vectors(lang):
    tv = TRANSLATIONS[lang]["threat_vectors"]
    localized = {}
    for key, info in tv.items():
        localized[key] = {
            "label": info["label"],
            "desc": info["desc"],
            "color": THREAT_VECTORS_BASE[key]["color"],
            "bg": THREAT_VECTORS_BASE[key]["bg"],
            "keywords": THREAT_VECTORS_BASE[key]["keywords"],
        }
    return localized

THREAT_VECTORS_BASE = {
    "IBAN_PHISHING": {"color": "#dc2626", "bg": "#fef2f2",
                      "keywords": ["iban", "broj računa", "prenos", "uplata na račun"]},
    "URGENCY": {"color": "#d97706", "bg": "#fffbeb",
                "keywords": ["hitno", "rok", "odmah", "danas", "blokiran", "isteklo", "kazna", "blokirajte"]},
    "CREDENTIAL_HARVEST": {"color": "#7c3aed", "bg": "#f5f3ff",
                           "keywords": ["pin", "lozinka", "jmbg", "sms kod", "verifikujte", "potvrdite", "unesite"]},
    "FAKE_DELIVERY": {"color": "#0891b2", "bg": "#ecfeff",
                      "keywords": ["paket", "carinu", "doplata", "pošta", "dostava", "pošiljalac", "preuzmite"]},
    "VISHING": {"color": "#be185d", "bg": "#fdf2f8",
                "keywords": ["0800", "pozovite", "nazovite", "besplatan poziv", "call center"]},
    "PRIZE_SCAM": {"color": "#15803d", "bg": "#f0fdf4",
                   "keywords": ["nagrada", "osvojili", "besplatno", "poklon", "izabrani ste", "dobitnik"]},
    "FINANCIAL_BAIT": {"color": "#b45309", "bg": "#fffbeb",
                       "keywords": ["refundacija", "povrat", "rata", "naknadu", "kredit", "zajam"]},
    "TYPOSQUATTING": {"color": "#991b1b", "bg": "#fef2f2", "keywords": []},
}

PHISHING_KEYWORDS = [
    "platite", "plati", "plaćanje", "unesite", "potvrdite", "verifikujte",
    "blokirajte", "hitno", "rok", "ažurirajte", "preuzmite", "aktivirajte",
    "bit.ly", ".net", ".info", ".help", ".xyz", "nagrada", "osvojili",
    "besplatno", "kazna", "carinu", "doplata", "jmbg", "pin", "sms kod",
    "iban", "0800", "pozovite", "nazovite", "naknadu", "refundaciju", "rata",
]

REPORT_CONTACTS = {
    "sr": {"cert_name": "CERT Srbije", "cert_url": "https://www.cert.rs/rs/prijava.html",
           "police_name": "MUP / OVTK", "police_url": "https://www.mup.gov.rs/wps/portal/sr/kontakt"},
    "hr": {"cert_name": "CERT.hr", "cert_url": "https://www.cert.hr/oincprijavi/",
           "police_name": "MUP RH", "police_url": "https://www.gov.hr/mup"},
    "bs": {"cert_name": "CERT.ba", "cert_url": "https://cert.ba/bs/contact",
           "police_name": "MUP BiH", "police_url": "https://www.mup.gov.ba"},
    "me": {"cert_name": "CERT.me", "cert_url": "https://www.cert.me/kontakt",
           "police_name": "MUP CG", "police_url": "https://www.gov.me/mpb"},
}

# =========================
# HELPER FUNCTIONS
# =========================
def get_red_flags(text):
    text_lower = text.lower()
    return [kw for kw in PHISHING_KEYWORDS if kw in text_lower]

def get_threat_vectors(text, lang):
    text_lower = text.lower()
    tv = get_localized_threat_vectors(lang)
    detected = []
    for key, info in tv.items():
        if key == "TYPOSQUATTING":
            continue
        if any(kw in text_lower for kw in info["keywords"]):
            detected.append(key)
    return detected

def extract_domains(text):
    url_pattern = r'https?://[^\s<>"{}|^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text.lower())
    domains = []
    for url in urls:
        domain = url.replace('http://', '').replace('https://', '').replace('www.', '')
        domain = domain.split('/')[0].split('?')[0]
        domains.append(domain)
    return domains

def check_typosquatting(text):
    domains = extract_domains(text)
    return [d for d in domains if d in ALL_SUSPICIOUS_DOMAINS]

def analyze_sms_length(text, lang):
    length = len(text)
    gsm_chars = set('@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !"#¤%&\'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà')
    is_gsm = all(c in gsm_chars for c in text)
    max_length = 160 if is_gsm else 70
    t = TRANSLATIONS[lang]
    if length < 20:
        return length, t["len_too_short"], "#ef4444", t["len_too_short_desc"]
    elif length < 60:
        return length, t["len_suspicious"], "#f59e0b", t["len_suspicious_desc"]
    elif length > max_length * 2:
        return length, t["len_too_long"], "#f59e0b", t["len_too_long_desc"].format(max_len=max_length)
    return length, t["len_std"], "#10b981", t["len_std_desc"]

def infer_language(t):
    t_lower = t.lower()
    if any(w in t_lower for w in ["hrvatska", "hpb", "fina", "pošta hrvatska"]):
        return "hr"
    if any(w in t_lower for w in ["bosna", "fbih", "bih", "bosanski", "pošta bih"]):
        return "bs"
    if any(w in t_lower for w in ["crna gora", "podgorica", "telekom cg"]):
        return "me"
    if re.search(r'[\u0400-\u04FF]', t):
        return "sr"
    return "sr"

# =========================
# RENDER HELPERS
# =========================
def render_confidence_bar(confidence, status_color, lang):
    t = TRANSLATIONS[lang]
    bar_width = max(0, min(100, confidence)) * 2.4
    return f"""
    <div style="margin:8px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:11px;color:#64748b;font-weight:600;">📊 {t['ensemble_confidence']}</span>
            <span style="font-size:13px;font-weight:700;color:{status_color};">{confidence}%</span>
        </div>
        <div style="background:#e2e8f0;border-radius:999px;height:10px;overflow:hidden;">
            <div style="width:{bar_width}px;max-width:100%;height:100%;background:linear-gradient(90deg,{status_color}99,{status_color});border-radius:999px;"></div>
        </div>
    </div>
    """

def render_flag_chips(threat_keys, typosquat_domains, lang):
    if not threat_keys and not typosquat_domains:
        return ""
    tv = get_localized_threat_vectors(lang)
    chips = []
    for key in threat_keys:
        v = tv[key]
        chips.append(
            f'<span title="{v["desc"]}" style="display:inline-flex;align-items:center;gap:4px;background:{v["bg"]};color:{v["color"]};border:1px solid {v["color"]}33;border-radius:999px;padding:3px 10px;font-size:11px;font-weight:600;cursor:help;white-space:nowrap;">{v["label"]}</span>'
        )
    if typosquat_domains:
        v = tv["TYPOSQUATTING"]
        for d in typosquat_domains:
            chips.append(
                f'<span title="{v["desc"]}: {d}" style="display:inline-flex;align-items:center;gap:4px;background:{v["bg"]};color:{v["color"]};border:1px solid {v["color"]}33;border-radius:999px;padding:3px 10px;font-size:11px;font-weight:600;cursor:help;white-space:nowrap;">{v["label"]}: {d}</span>'
            )
    return '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;">' + " ".join(chips) + '</div>'

def render_ensemble_row(pred_a, conf_a, pred_b, conf_b, b_available, lang):
    t = TRANSLATIONS[lang]
    if not b_available:
        return f"""
        <div style="font-size:11px;color:#64748b;margin:4px 0;padding:6px 10px;background:#f8fafc;border-radius:8px;">
        {t['model_tfidf']} <strong>{pred_a}</strong> ({conf_a}%) | 
        {t['model_bertic']} <span style="color:#94a3b8;">{t['model_unavailable']}</span>
        </div>
        """
    return f"""
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;font-size:11px;color:#64748b;margin:4px 0;padding:6px 10px;background:#f8fafc;border-radius:8px;">
        <span>{t['model_tfidf']} <strong>{pred_a}</strong> {conf_a}%</span>
        <span>{t['model_bertic']} <strong>{pred_b}</strong> {conf_b}%</span>
    </div>
    """

# =========================
# MAIN DETECTION FUNCTION
# =========================
def detect_sms(text, is_human, agrees_privacy, lang):
    t = TRANSLATIONS[lang]
    if not is_human:
        return f"<div style='background:#fef3c7;padding:8px 12px;border-radius:8px;border-left:3px solid #f59e0b;color:#92400e;font-size:13px;'>️ Please confirm you are not a robot</div>" if lang == "en" else "<div style='background:#fef3c7;padding:8px 12px;border-radius:8px;border-left:3px solid #f59e0b;color:#92400e;font-size:13px;'>️ Molimo potvrdite da niste robot</div>"
    if agrees_privacy is False or agrees_privacy is None or agrees_privacy == 0:
        return f"<div style='background:#fef3c7;padding:8px 12px;border-radius:8px;border-left:3px solid #f59e0b;color:#92400e;font-size:13px;'> You must accept the terms of use</div>" if lang == "en" else "<div style='background:#fef3c7;padding:8px 12px;border-radius:8px;border-left:3px solid #f59e0b;color:#92400e;font-size:13px;'> Morate se složiti sa uslovima korišćenja</div>"
    
    text = text.strip()
    if not text:
        return f"<div style='background:#f8fafc;padding:8px 12px;border-radius:8px;border-left:3px solid #64748b;color:#475569;font-size:13px;'>⬅️ Enter an SMS message</div>" if lang == "en" else "<div style='background:#f8fafc;padding:8px 12px;border-radius:8px;border-left:3px solid #64748b;color:#475569;font-size:13px;'>⬅️ Unesite SMS poruku</div>"
    
    length, len_status, len_color, len_desc = analyze_sms_length(text, lang)
    red_flags = get_red_flags(text)
    typosquat_domains = check_typosquatting(text)
    threat_keys = get_threat_vectors(text, lang)
    lang_key = infer_language(text)
    
    if tfidf_model is not None:
        pred_a = tfidf_model.predict([text])[0]
        conf_a = round(max(tfidf_model.predict_proba([text])[0]) * 100, 2)
    else:
        pred_a = "legitimate"
        conf_a = 50.0
    
    if BERTIC_AVAILABLE:
        bertic_result = predict_with_bertic(text, lang_key)
        pred_b = bertic_result["label"].lower()
        conf_b = round(bertic_result["confidence"] * 100, 2)
    else:
        pred_b = conf_b = None
    
    if typosquat_domains:
        prediction = "phishing"
        confidence = 95.0
    elif pred_b is not None:
        prediction = pred_b
        confidence = conf_b
    else:
        prediction = pred_a
        confidence = conf_a
    
    models_agree = (pred_b is None) or (pred_a == pred_b)
    THRESHOLD_HIGH, THRESHOLD_LOW = 80, 60
    
    if typosquat_domains or (prediction == "phishing" and confidence >= THRESHOLD_HIGH):
        status_line = t["status_phishing_high"]
        status_color = "#dc2626"
        status_bg = "#fef2f2"
        why_parts = [t["why_both_phishing"] if (BERTIC_AVAILABLE and models_agree) else t["why_model_detected"]]
        if red_flags:
            why_parts.append(t["why_red_flags"].format(red_flags=", ".join(red_flags)))
        if typosquat_domains:
            why_parts.append(t["why_typosquatting"].format(domains=", ".join(typosquat_domains)))
        action = t["action_phishing_high"]
    elif THRESHOLD_LOW <= confidence < THRESHOLD_HIGH:
        status_line = t["status_not_sure"]
        status_color = "#d97706"
        status_bg = "#fffbeb"
        why_parts = [t["why_low_confidence"]]
        if not models_agree:
            why_parts.append(t["why_disagree_tfidf"].format(pred_a=pred_a, pred_b=pred_b))
        if red_flags:
            why_parts.append(t["why_red_flags"].format(red_flags=", ".join(red_flags)))
        action = t["action_not_sure"]
    elif prediction == "phishing":
        status_line = t["status_possible_fraud"]
        status_color = "#d97706"
        status_bg = "#fffbeb"
        why_parts = [t["why_suspicious_features"]]
        if red_flags:
            why_parts.append(t["why_red_flags"].format(red_flags=", ".join(red_flags)))
        action = t["action_possible_fraud"]
    else:
        status_line = t["status_legit"]
        status_color = "#059669"
        status_bg = "#f0fdf4"
        why_parts = [t["why_legit"]]
        if not models_agree:
            why_parts.append(t["why_disagree_tfidf"].format(pred_a=pred_a, pred_b=pred_b))
        if red_flags:
            why_parts.append(t["why_note_red_flags"].format(red_flags=", ".join(red_flags)))
        action = t["action_legit"]
    
    contacts = REPORT_CONTACTS.get(lang_key, REPORT_CONTACTS["sr"])
    is_danger = (prediction == "phishing") or (not models_agree) or (THRESHOLD_LOW <= confidence < THRESHOLD_HIGH)
    
    action_buttons = ""
    if is_danger:
        warning_msg = TRANSLATIONS[lang]["warning_text"]
        escaped_text = text.replace('"', "'").replace("\n", " ")
        action_buttons = f"""
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #e2e8f0;">
            <p style="margin:0 0 6px 0;font-weight:bold;color:#334155;font-size:12px;">{t['what_to_do']}</p>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
                <button onclick="navigator.clipboard.writeText(this.dataset.txt).then(() => this.textContent='{t['copy_success']}').catch(() => this.textContent='{t['copy_error']}')" data-txt="{escaped_text}"
                  style="background:#64748b;color:white;border:none;border-radius:6px;padding:5px 10px;font-size:11px;cursor:pointer;font-weight:bold;">{t['copy_btn']}</button>
                <a href="{contacts['cert_url']}" target="_blank" style="background:#b91c1c;color:white;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:bold;text-decoration:none;display:inline-block;">{t['report_cert_btn'].format(name=contacts['cert_name'])}</a>
                <a href="{contacts['police_url']}" target="_blank" style="background:#2563eb;color:white;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:bold;text-decoration:none;display:inline-block;">{t['report_police_btn'].format(name=contacts['police_name'])}</a>
                <a href="sms:?body={warning_msg}" style="background:#059669;color:white;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:bold;text-decoration:none;display:inline-block;">{t['warn_contact_btn']}</a>
            </div>
        </div>
        """
    
    flag_chips = render_flag_chips(threat_keys, typosquat_domains, lang)
    confidence_bar = render_confidence_bar(confidence, status_color, lang)
    ensemble_row = render_ensemble_row(pred_a, conf_a, pred_b, conf_b, BERTIC_AVAILABLE, lang)
    length_label = t["length_label"].format(length=length)
    
    result_html = f"""
    <div style="font-family:'Segoe UI',sans-serif;font-size:13px;background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="background:{status_bg};border:1.5px solid {status_color}33;border-radius:10px;padding:10px 14px;margin-bottom:12px;text-align:center;">
            <p style="font-size:17px;font-weight:800;margin:0 0 2px 0;color:{status_color};">{status_line}</p>
        </div>
        {confidence_bar}
        {ensemble_row}
        {f'<div style="margin:8px 0;"><p style="font-size:11px;font-weight:600;color:#64748b;margin:0 0 4px 0;">{t["detected_vectors"]}</p>{flag_chips}</div>' if flag_chips else ''}
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:7px 12px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#64748b;font-size:11px;">{length_label}</span>
                <span style="background:{len_color};color:white;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600;">{len_status}</span>
            </div>
            <p style="margin:2px 0 0 0;font-size:10px;color:#94a3b8;">{len_desc}</p>
        </div>
        <div style="border-left:3px solid {status_color};padding:8px 12px;border-radius:0 8px 8px 0;background:#fafafa;">
            <p style="margin:0 0 4px 0;font-weight:700;color:#1e293b;font-size:12px;">{t['why_label']}</p>
            <p style="margin:0 0 8px 0;color:#475569;font-size:12px;line-height:1.5;">{"<br>".join(why_parts)}</p>
            <p style="margin:0 0 2px 0;font-weight:700;color:#1e293b;font-size:12px;">{t['advice_label']}</p>
            <p style="margin:0;color:#475569;font-size:12px;line-height:1.5;">{action}</p>
            {action_buttons}
        </div>
    </div>
    """
    return result_html

# =========================
# BATCH TEST
# =========================
def run_batch(file, lang):
    t = TRANSLATIONS[lang]
    if file is None:
        return f"<p style='color:#94a3b8;'>Upload a CSV file.</p>" if lang == "en" else "<p style='color:#94a3b8;'>Učitajte CSV fajl.</p>"
    try:
        df_test = pd.read_csv(file.name)
    except Exception as e:
        return f"<p style='color:#dc2626;'>Error: {e}</p>"
    
    if not {'text', 'label'}.issubset(set(df_test.columns)):
        return f"<p style='color:#dc2626;'>CSV must have columns: <strong>text</strong> and <strong>label</strong></p>" if lang == "en" else "<p style='color:#dc2626;'>CSV mora imati kolone: <strong>text</strong> i <strong>label</strong></p>"
    
    THRESHOLD_HIGH, THRESHOLD_LOW = 80, 60
    results, correct, uncertain, total = [], 0, 0, len(df_test)
    
    for _, row in df_test.iterrows():
        text = str(row.get('text', '')).strip()
        expected = str(row.get('label', '')).strip().lower()
        technique = str(row.get('technique', '—'))
        if not text:
            continue
        
        lang_key = infer_language(text)
        if tfidf_model is not None:
            pred_a = tfidf_model.predict([text])[0]
            conf_a = round(max(tfidf_model.predict_proba([text])[0]) * 100, 2)
        else:
            pred_a, conf_a = "legitimate", 50.0
        if BERTIC_AVAILABLE:
            bertic_result = predict_with_bertic(text, lang_key)
            pred_b, conf_b = bertic_result["label"].lower(), round(bertic_result["confidence"] * 100, 2)
        else:
            pred_b, conf_b = None, None
        
        typosquat = check_typosquatting(text)
        models_agree = (pred_b is None) or (pred_a == pred_b)
        if typosquat:
            prediction, confidence = "phishing", 95.0
        elif pred_b is not None:
            prediction, confidence = pred_b, conf_b
        else:
            prediction, confidence = pred_a, conf_a
        
        threat_keys = get_threat_vectors(text, lang)
        tv_local = get_localized_threat_vectors(lang)
        flag_mini = " ".join(
            f'<span style="background:{tv_local[k]["bg"]};color:{tv_local[k]["color"]};border-radius:999px;padding:1px 6px;font-size:10px;font-weight:600;white-space:nowrap;margin-right:4px;">{tv_local[k]["label"]}</span>'
            for k in threat_keys[:2]
        )
        if typosquat:
            v = tv_local["TYPOSQUATTING"]
            flag_mini += f'<span style="background:{v["bg"]};color:{v["color"]};border-radius:999px;padding:1px 6px;font-size:10px;font-weight:600;">🌐 Typosquatting</span>'
        
        if typosquat or (prediction == "phishing" and confidence >= THRESHOLD_HIGH):
            verdict, verdict_label, verdict_color = "phishing", t["batch_phishing_label"], "#dc2626"
        elif THRESHOLD_LOW <= confidence < THRESHOLD_HIGH:
            verdict, verdict_label, verdict_color = "uncertain", t["batch_uncertain_label"], "#d97706"
            uncertain += 1
        elif prediction == "phishing":
            verdict, verdict_label, verdict_color = "phishing", t["batch_possible_label"], "#d97706"
        else:
            verdict, verdict_label, verdict_color = "legitimate", t["batch_legit_label"], "#059669"
        
        is_correct = (verdict != "uncertain") and ((verdict == "phishing" and expected == "phishing") or (verdict == "legitimate" and expected == "legitimate"))
        if is_correct:
            correct += 1
        row_bg = "#f0fdf4" if is_correct else ("#fefce8" if verdict == "uncertain" else "#fef2f2")
        icon = "✅" if is_correct else ("⚠️" if verdict == "uncertain" else "❌")
        ensemble_mini = f"A:{pred_a[:1].upper()}/{conf_a}%" + (f" B:{pred_b[:1].upper()}/{conf_b}%" if pred_b else " B:—")
        
        results.append(f"""
        <tr style="background:{row_bg};border-bottom:1px solid #e2e8f0;">
            <td style="padding:4px 8px;font-size:11px;color:#64748b;">{technique}</td>
            <td style="padding:4px 8px;font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{text}">{text[:45]}{"..." if len(text) > 45 else ""}</td>
            <td style="padding:4px 8px;font-size:11px;text-align:center;"><strong>{expected}</strong></td>
            <td style="padding:4px 8px;font-size:11px;text-align:center;color:{verdict_color};font-weight:bold;">{verdict_label}</td>
            <td style="padding:4px 8px;font-size:10px;text-align:center;color:#0ea5e9;">{ensemble_mini}</td>
            <td style="padding:4px 8px;font-size:12px;font-weight:500;">{flag_mini}</td>
            <td style="padding:4px 8px;font-size:11px;text-align:center;">{icon}</td>
        </tr>
        """)
    
    accuracy = round(correct / (total - uncertain) * 100, 1) if (total - uncertain) > 0 else 0
    summary = f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin-bottom:12px;">
        <p style="margin:0 0 6px 0;font-size:14px;font-weight:bold;color:#0e7490;">{t['batch_summary']}</p>
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
            <span style="font-size:12px;">{t['batch_total']} <strong>{total}</strong></span>
            <span style="font-size:12px;color:#059669;">{t['batch_correct']} <strong>{correct}</strong></span>
            <span style="font-size:12px;color:#dc2626;">{t['batch_wrong']} <strong>{total - correct - uncertain}</strong></span>
            <span style="font-size:12px;color:#d97706;">{t['batch_uncertain']} <strong>{uncertain}</strong></span>
            <span style="font-size:12px;color:#0ea5e9;">{t['batch_accuracy']} <strong>{accuracy}%</strong></span>
        </div>
    </div>
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;">
        <thead><tr style="background:#0e7490;color:white;">
            <th style="padding:6px 8px;font-size:11px;text-align:left;">{t['batch_technique']}</th>
            <th style="padding:6px 8px;font-size:11px;text-align:left;">{t['batch_message']}</th>
            <th style="padding:6px 8px;font-size:11px;text-align:center;">{t['batch_expected']}</th>
            <th style="padding:6px 8px;font-size:11px;text-align:center;">{t['batch_verdict']}</th>
            <th style="padding:6px 8px;font-size:11px;text-align:center;">{t['batch_models']}</th>
            <th style="padding:6px 8px;font-size:11px;text-align:left;">{t['batch_vectors']}</th>
            <th style="padding:6px 8px;font-size:11px;text-align:center;">{t['batch_ok']}</th>
        </tr></thead>
        <tbody>{"".join(results)}</tbody>
    </table>
    </div>
    """
    return summary

# =========================
# UI UPDATE FUNCTION
# =========================
def update_ui(lang):
    t = TRANSLATIONS[lang]
    quick_tests_html = f'<div style="text-align:center; font-weight:bold; font-size:18px; margin-top:8px;">{t["quick_tests"]}</div>'
    return (
        gr.update(value=t["title"]),
        gr.update(value=t["subtitle"]),
        gr.update(value=f'<div class="sms-label">{t["sms_input_label"]}</div>'),
        gr.update(placeholder=t["sms_placeholder"]),
        gr.update(label=t["human_check"]),
        gr.update(label=t["privacy_check"]),
        gr.update(value=t["analyze_btn"]),
        gr.update(value=f'<p style="margin:0 0 6px 0;font-size:13px;font-weight:600;color:#1e3a5f;">{t["output_label"]}</p>'),
        gr.update(value=" "),
        gr.update(value=quick_tests_html),
        gr.update(value=t["phishing_title"]),
        gr.update(value=t["legit_title"]),
        gr.update(value=t["batch_title"]),
        gr.update(value=t["batch_desc"]),
        gr.update(label=t["upload_csv"]),
        gr.update(value=t["run_batch"]),
        gr.update(value=t["batch_output"]),
        gr.update(value=t["footer"]),
        gr.update(value=t["ex_phishing_1"][:40] + "..."),
        gr.update(value=t["ex_phishing_2"][:40] + "..."),
        gr.update(value=t["ex_legit_1"][:40] + "..."),
        gr.update(value=t["ex_legit_2"][:40] + "..."),
    )

def get_example(lang, example_key):
    return TRANSLATIONS[lang][example_key]

# =========================
# CSS (fixed alignment and styling)
# =========================
css = """
footer { display: none !important; }
body, .gradio-container { background: #f0f4f8 !important; font-family: 'Segoe UI', Arial, sans-serif; }
h1, h2, h3, .gr-markdown h1 { color: #1e3a5f !important; text-align: center !important; margin: 0 0 6px 0 !important; }
.gr-markdown h1, .gr-markdown h1:first-child {
    font-size: 34px !important;
    font-weight: 900 !important;
    text-align: center !important;
    letter-spacing: -0.5px;
    margin: 0 0 12px 0 !important;
}
.gr-markdown p { text-align: center !important; margin: 0 0 8px 0 !important; color: #475569 !important; }

/* Make the main row stretch to equal height */
.main-row {
    display: flex !important;
    align-items: stretch !important;
}

/* Left and right panels: make them flex containers */
#left-panel, #right-panel {
    background: #f1f5f9 !important;
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 16px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    height: 100% !important;
}

/* FIX: Align top elements perfectly */
#left-panel .form, #left-panel .wrap-input { margin-top: 0 !important; }
#right-panel .gr-html { margin-top: 0 !important; }

/* Ensure the output area expands and has min height */
#right-panel .gr-html {
    flex: 1 !important;
    min-height: 280px !important;
}

.gradio-container .form textarea, .gradio-container .form .prose, .gradio-container .form .output-wrap {
    background: #ffffff !important; border: 1.5px solid #cbd5e1 !important; border-radius: 12px !important;
    color: #1e293b !important; font-size: 13px !important; padding: 12px !important;
}
#left-panel .checkbox-container,
#left-panel .form .checkbox-container {
    margin: 2px 0 !important;
    padding: 2px 0 !important;
    display: block !important;
    background: #f1f5f9 !important;
}
#left-panel .checkbox-container label,
#left-panel .form .checkbox-container label {
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
}
button.primary {
    background: #1e3a5f !important; color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 16px !important; font-weight: 700 !important; padding: 12px 32px !important; min-width: 180px !important;
    margin: 12px auto !important; display: block !important; cursor: pointer !important;
}
button.primary:hover { background: #2563eb !important; transform: translateY(-2px) !important; }
table td:nth-child(6), table th:nth-child(6) { font-size: 12px !important; font-weight: 500 !important; }
.top-bar { display: flex; justify-content: flex-end; margin-bottom: 8px; }
.sms-label { font-size: 14px; font-weight: 600; color: #1e3a5f; margin-bottom: 4px; display: block; }
.gradio-container .form .row .gr-form-row { gap: 0 !important; }
"""

# =========================
# UI CONSTRUCTION
# =========================
with gr.Blocks(css=css, title="Ne Nasedaj 🛡️") as demo:
    with gr.Row(elem_classes="top-bar"):
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=0, min_width=200):
            lang_radio = gr.Radio(choices=[("🇬🇧 EN", "en"), ("🇷🇸 SR", "sr")], label="🌐 Language / Jezik", value="sr", interactive=True)
    
    title_md = gr.Markdown(TRANSLATIONS["sr"]["title"])
    subtitle_md = gr.Markdown(TRANSLATIONS["sr"]["subtitle"])
    
    with gr.Row(equal_height=True, elem_classes="main-row"):
        with gr.Column(scale=1, elem_id="left-panel"):
            sms_label = gr.HTML(f'<div class="sms-label">{TRANSLATIONS["sr"]["sms_input_label"]}</div>')
            sms_input = gr.Textbox(label=None, show_label=False, lines=6, placeholder=TRANSLATIONS["sr"]["sms_placeholder"])
            human_check = gr.Checkbox(label=TRANSLATIONS["sr"]["human_check"], value=False)
            privacy_check = gr.Checkbox(label=TRANSLATIONS["sr"]["privacy_check"], value=False)
            terms_html = gr.HTML(f"""
            <div style="margin:-4px 0 4px 0;">
                <a href="#" onclick="document.getElementById('terms-modal').style.display='flex';return false;" style="font-size:11px;color:#0e7490;text-decoration:underline;cursor:pointer;">📄 {TRANSLATIONS['sr']['terms_link']}</a>
            </div>
            <div id="terms-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;align-items:center;justify-content:center;">
                <div style="background:white;border-radius:12px;padding:24px;max-width:480px;width:90%;max-height:80vh;overflow-y:auto;position:relative;">
                    <button onclick="document.getElementById('terms-modal').style.display='none'" style="position:absolute;top:12px;right:16px;background:none;border:none;font-size:20px;cursor:pointer;">✕</button>
                    <h3 style="margin:0 0 12px 0;color:#0e7490;">{TRANSLATIONS['sr']['terms_title']}</h3>
                    <p style="font-size:12px;">{TRANSLATIONS['sr']['terms_p1']}</p><p style="font-size:12px;">{TRANSLATIONS['sr']['terms_p2']}</p>
                    <p style="font-size:12px;">{TRANSLATIONS['sr']['terms_p3']}</p><p style="font-size:12px;">{TRANSLATIONS['sr']['terms_p4']}</p>
                    <p style="font-size:12px;">{TRANSLATIONS['sr']['terms_p5']}</p>
                    <button onclick="document.getElementById('terms-modal').style.display='none'" style="background:#0e7490;color:white;border:none;border-radius:8px;padding:8px 20px;width:100%;">{TRANSLATIONS['sr']['terms_accept']}</button>
                </div>
            </div>
            """)
            analyze_btn = gr.Button(TRANSLATIONS["sr"]["analyze_btn"], variant="primary")
        
        with gr.Column(scale=1, elem_id="right-panel"):
            output_label = gr.HTML(f'<p style="margin:0 0 6px 0;font-size:13px;font-weight:600;color:#1e3a5f;">{TRANSLATIONS["sr"]["output_label"]}</p>')
            output = gr.HTML(value=" ")
    
    # Quick tests
    quick_tests_html = f'<div style="text-align:center; font-weight:bold; font-size:18px; margin-top:8px;">{TRANSLATIONS["sr"]["quick_tests"]}</div>'
    quick_tests_md = gr.HTML(quick_tests_html)
    hr_before = gr.HTML("<hr style='margin:16px 0;'>")
    
    # Example titles above columns
    with gr.Row():
        with gr.Column():
            phishing_title_md = gr.Markdown(f"**{TRANSLATIONS['sr']['phishing_title']}**")
            ex_btn1 = gr.Button(TRANSLATIONS["sr"]["ex_phishing_1"][:40] + "...", size="sm")
            ex_btn2 = gr.Button(TRANSLATIONS["sr"]["ex_phishing_2"][:40] + "...", size="sm")
        with gr.Column():
            legit_title_md = gr.Markdown(f"**{TRANSLATIONS['sr']['legit_title']}**")
            ex_btn3 = gr.Button(TRANSLATIONS["sr"]["ex_legit_1"][:40] + "...", size="sm")
            ex_btn4 = gr.Button(TRANSLATIONS["sr"]["ex_legit_2"][:40] + "...", size="sm")
    
    # Batch test section
    batch_title_md = gr.Markdown(f"---\n### {TRANSLATIONS['sr']['batch_title']}")
    batch_desc_md = gr.Markdown(TRANSLATIONS["sr"]["batch_desc"])
    with gr.Row():
        csv_upload = gr.File(label=TRANSLATIONS["sr"]["upload_csv"], file_types=[".csv"])
        run_batch_btn = gr.Button(TRANSLATIONS["sr"]["run_batch"], variant="primary")
    batch_output = gr.HTML(label=TRANSLATIONS["sr"]["batch_output"])
    footer_md = gr.Markdown(f"---\n{TRANSLATIONS['sr']['footer']}")
    
    # Language change handlers
    lang_radio.change(
        fn=update_ui,
        inputs=lang_radio,
        outputs=[
            title_md, subtitle_md, sms_label, sms_input, human_check, privacy_check, analyze_btn, output_label, output,
            quick_tests_md, phishing_title_md, legit_title_md, batch_title_md, batch_desc_md,
            csv_upload, run_batch_btn, batch_output, footer_md,
            ex_btn1, ex_btn2, ex_btn3, ex_btn4
        ]
    )
    
    def update_terms(lang):
        t = TRANSLATIONS[lang]
        return f"""
        <div style="margin:-4px 0 4px 0;"><a href="#" onclick="document.getElementById('terms-modal').style.display='flex';return false;" style="font-size:11px;color:#0e7490;text-decoration:underline;cursor:pointer;">📄 {t['terms_link']}</a></div>
        <div id="terms-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;align-items:center;justify-content:center;">
            <div style="background:white;border-radius:12px;padding:24px;max-width:480px;width:90%;max-height:80vh;overflow-y:auto;position:relative;">
                <button onclick="document.getElementById('terms-modal').style.display='none'" style="position:absolute;top:12px;right:16px;background:none;border:none;font-size:20px;cursor:pointer;">✕</button>
                <h3 style="margin:0 0 12px 0;color:#0e7490;">{t['terms_title']}</h3>
                <p style="font-size:12px;">{t['terms_p1']}</p><p style="font-size:12px;">{t['terms_p2']}</p>
                <p style="font-size:12px;">{t['terms_p3']}</p><p style="font-size:12px;">{t['terms_p4']}</p>
                <p style="font-size:12px;">{t['terms_p5']}</p>
                <button onclick="document.getElementById('terms-modal').style.display='none'" style="background:#0e7490;color:white;border:none;border-radius:8px;padding:8px 20px;width:100%;">{t['terms_accept']}</button>
            </div>
        </div>
        """
    
    lang_radio.change(fn=update_terms, inputs=lang_radio, outputs=terms_html)
    
    # Example buttons insert full text
    ex_btn1.click(fn=lambda lang: get_example(lang, "ex_phishing_1"), inputs=lang_radio, outputs=sms_input)
    ex_btn2.click(fn=lambda lang: get_example(lang, "ex_phishing_2"), inputs=lang_radio, outputs=sms_input)
    ex_btn3.click(fn=lambda lang: get_example(lang, "ex_legit_1"), inputs=lang_radio, outputs=sms_input)
    ex_btn4.click(fn=lambda lang: get_example(lang, "ex_legit_2"), inputs=lang_radio, outputs=sms_input)
    
    analyze_btn.click(fn=detect_sms, inputs=[sms_input, human_check, privacy_check, lang_radio], outputs=output)
    run_batch_btn.click(fn=run_batch, inputs=[csv_upload, lang_radio], outputs=batch_output)

if __name__ == "__main__":
    demo.launch()
