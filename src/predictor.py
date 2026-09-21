import re
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# Model Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "civicsense_indicbert"

# ============================================================
# Load Model
# ============================================================

MODEL_LOAD_ERROR = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model folder not found at: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    id2label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

except Exception as error:
    MODEL_LOAD_ERROR = str(error)
    tokenizer = None
    model = None
    device = None
    id2label = {}


# ============================================================
# Domain Keywords
# ============================================================

keyword_rules = {

    "Electricity": [
        # English
        "electricity", "electric meter", "meter reading", "power cut",
        "power outage", "power supply", "transformer", "voltage",
        "electric bill", "electric pole",
        # Roman Hindi
        "bijli", "bijli nahi", "bijli gayi", "bijli band", "light nahi",
        "light gayi", "karant", "meter", "bijli bill",
        # Hindi
        "बिजली", "बिजली नहीं", "बिजली गई", "बिजली बंद", "लाइट नहीं",
        "लाइट गई", "करंट", "मीटर", "बिजली बिल", "ट्रांसफार्मर", "वोल्टेज"
    ],

    "Water Supply": [
        # English
        "water supply", "drinking water", "water shortage", "water tanker",
        "water pressure", "water leak", "pipeline",
        # Roman Hindi
        "paani", "pani", "paani nahi", "pani nahi", "nal", "nal ka paani",
        "paani ki supply",
        # Hindi
        "पानी", "पानी नहीं", "पानी की समस्या", "पानी की सप्लाई", "पीने का पानी",
        "जल आपूर्ति", "जल समस्या", "नल", "नल का पानी", "पाइपलाइन", "पानी की कमी", "टैंकर"
    ],

    "Land Records & Revenue": [
        # English
        "land record", "land records", "khasra", "khasra number", "jamabandi",
        "mutation", "property transfer",
        # Roman Hindi
        "patwari", "zameen", "zameen ka record", "bhumi", "jameen", "registry",
        # Hindi
        "पटवारी", "जमीन", "भूमि", "भूमि रिकॉर्ड", "जमीन का रिकॉर्ड", "खसरा",
        "खसरा नंबर", "जमाबंदी", "नामांतरण", "म्यूटेशन", "रजिस्ट्री", "संपत्ति"
    ],

    "Sanitation & Garbage": [
        # English
        "garbage", "waste collection", "public toilet", "toilet smell", "dirty toilet",
        "ganda toilet", "foul smell", "sewer", "sewage", "drain cleaning",
        # Roman Hindi
        "kachra", "safai", "gandagi", "ganda", "gandi smell", "kachra nahi uthaya",
        # Hindi
        "कचरा", "सफाई", "गंदगी", "गंदा", "गंदी बदबू", "बदबू", "कूड़ा",
        "कूड़ा नहीं उठाया", "शौचालय", "नाली", "सीवर", "गंदा पानी"
    ],

    "Municipal Certificates": [
        # English
        "death certificate", "birth certificate", "municipal certificate",
        # Roman Hindi
        "janam praman patra", "mrityu praman patra", "janam certificate", "mrityu certificate",
        # Hindi
        "जन्म प्रमाण पत्र", "मृत्यु प्रमाण पत्र", "जन्म प्रमाणपत्र", "मृत्यु प्रमाणपत्र",
        "नगरपालिका प्रमाण पत्र", "प्रमाण पत्र"
    ],

    "Banking & Financial Services": [
        # English
        "bank account", "bank branch", "passbook", "atm", "bank transaction", "bank", "account",
        # Roman Hindi
        "khata", "bank khata", "khate", "paise", "transaction", "paise kat", "paisa kata",
        "account update",
        # Hindi
        "बैंक", "बैंक खाता", "खाता", "खाते", "खाते से पैसे", "पैसे कट", "पैसा कटा",
        "लेनदेन", "बैंक शाखा", "पासबुक", "एटीएम", "खाता अपडेट"
    ],

    "Education & Schools": [
        # English
        "government school", "school", "teacher", "student", "classroom", "school fees", "education",
        # Roman Hindi
        "vidyalaya", "shikshak", "padhai", "fees",
        # Hindi
        "स्कूल", "विद्यालय", "शिक्षक", "अध्यापक", "छात्र", "विद्यार्थी", "पढ़ाई", "शिक्षा",
        "कक्षा", "स्कूल फीस", "शुल्क"
    ],

    "Police & Law and Order": [
        # English
        "police station", "police", "fir",
        # Roman Hindi
        "thana", "thane", "thanedar", "report", "shikayat",
        # Hindi
        "पुलिस", "पुलिस स्टेशन", "थाना", "थाने", "थानेदार", "एफआईआर", "रिपोर्ट", "शिकायत", "मामला"
    ],

    "Pension & Provident Fund": [
        # English
        "pension", "provident fund", "pf", "pension payment", "retirement",
        # Roman Hindi
        "pension nahi", "pension nahi mili", "budhapa pension", "vriddha pension",
        # Hindi
        "पेंशन", "पेंशन नहीं", "पेंशन नहीं मिली", "पेंशन भुगतान", "वृद्धावस्था पेंशन",
        "विधवा पेंशन", "विकलांग पेंशन", "सेवानिवृत्ति", "भविष्य निधि", "पीएफ"
    ],

    "Ration & Public Distribution System": [
        # English
        "ration dealer", "ration card", "ration", "food grain", "public distribution",
        # Roman Hindi
        "anaj", "anaaj", "khadya", "sarkari ration", "ration dukaan",
        # Hindi
        "राशन", "राशन डीलर", "राशन कार्ड", "अनाज", "खाद्यान्न", "सरकारी राशन",
        "राशन दुकान", "सार्वजनिक वितरण", "उचित मूल्य की दुकान"
    ],

    "Healthcare & Hospitals": [
        # English
        "hospital", "doctor", "medicine", "blood bank", "blood", "treatment", "healthcare", "medical", "ambulance",
        # Roman Hindi
        "aspatal", "dawai", "dawa", "ilaaj", "sehat",
        # Hindi
        "अस्पताल", "हॉस्पिटल", "डॉक्टर", "दवाई", "दवा", "इलाज", "स्वास्थ्य", "चिकित्सा",
        "एम्बुलेंस", "रक्त", "ब्लड बैंक"
    ],

    "Employment & Labour": [
        # English
        "contract worker", "construction worker", "overtime", "employment", "worker", "labour", "labor",
        "salary", "wages", "job",
        # Roman Hindi
        "naukri", "rojgar", "mazdoor", "majdoor", "kaam", "vetan", "tankhwa",
        # Hindi
        "नौकरी", "रोजगार", "मजदूर", "श्रमिक", "कामगार", "काम", "वेतन", "तनख्वाह", "नौकरी नहीं", "श्रम"
    ],

    "Corruption & Bribery": [
        # English
        "bribe", "bribery", "corruption", "extra money",
        # Roman Hindi
        "rishwat", "rishvat", "ghoos", "bhrashtachar", "paise maang", "paise maang raha",
        "extra paisa", "extra paise",
        # Hindi
        "रिश्वत", "रिश्वत मांग", "रिश्वत मांग रहा", "घूस", "भ्रष्टाचार", "पैसे मांग",
        "पैसे मांग रहा", "अतिरिक्त पैसे", "ज्यादा पैसे"
    ],

    "Roads & Infrastructure": [
        # English
        "pothole", "potholes", "damaged road", "broken road", "street light", "street lights",
        "footpath", "flyover", "highway", "bridge", "road", "roads", "infrastructure",
        # Roman Hindi
        "sadak", "sadak kharab", "road kharab", "gaddha", "gaddhe", "pul", "footpath",
        # Hindi
        "सड़क", "सड़क खराब", "सड़क टूटी", "गड्ढा", "गड्ढे", "स्ट्रीट लाइट", "पुल", "फुटपाथ",
        "फ्लाईओवर", "हाईवे", "राजमार्ग", "सड़क की समस्या", "बुनियादी ढांचा"
    ]
}



# ============================================================
# Normalize Text
# ============================================================

def _normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# Keyword Matching
# ============================================================

def _keyword_match(text):

    text = _normalize_text(text)

    category_scores = {}
    matched_by_category = {}

    for category, keywords in keyword_rules.items():

        score = 0
        matches = []

        for keyword in keywords:

            keyword = keyword.lower().strip()

            # Unicode-safe substring matching. This supports both
            # Roman Hindi and Devanagari Hindi phrases.
            if keyword in text:

                matches.append(keyword)

                # More specific phrases get higher weight
                words = len(keyword.split())

                if words >= 3:
                    score += 3
                elif words == 2:
                    score += 2
                else:
                    score += 1

        category_scores[category] = score
        matched_by_category[category] = matches

    if not category_scores:
        return None, 0, []

    best_score = max(category_scores.values())

    if best_score == 0:
        return None, 0, []

    best_categories = [
        category
        for category, score in category_scores.items()
        if score == best_score
    ]

    # If keywords tie between categories,
    # don't blindly choose the first category.
    if len(best_categories) > 1:
        return None, best_score, []

    best_category = best_categories[0]

    return (
        best_category,
        best_score,
        matched_by_category[best_category]
    )


# ============================================================
# Hybrid Prediction
# ============================================================

def _weighted_hybrid_prediction(probabilities, complaint):

    text = _normalize_text(complaint)

    bert_id = torch.argmax(probabilities).item()
    bert_category = id2label[bert_id]
    bert_confidence = probabilities[bert_id].item()

    keyword_category, keyword_score, matched_keywords = (
        _keyword_match(text)
    )

    # --------------------------------------------------------
    # Case 1: No useful keyword evidence
    # --------------------------------------------------------

    if keyword_category is None:
        return (
            bert_category,
            bert_confidence,
            "IndicBERT",
            []
        )

    # --------------------------------------------------------
    # Case 2: BERT and keyword agree
    # --------------------------------------------------------

    if keyword_category == bert_category:
        return (
            bert_category,
            bert_confidence,
            "IndicBERT + Keyword Agreement",
            matched_keywords
        )

    # --------------------------------------------------------
    # Case 3: Strong, specific keyword evidence
    #
    # Only override BERT when keyword evidence is clearly
    # stronger than the model's low-confidence prediction.
    # --------------------------------------------------------

    if keyword_score >= 3 and bert_confidence < 0.45:
        return (
            keyword_category,
            bert_confidence,
            "Hybrid (Strong Keyword Match)",
            matched_keywords
        )

    # --------------------------------------------------------
    # Case 4: Moderate keyword evidence + very low confidence
    # --------------------------------------------------------

    if keyword_score >= 2 and bert_confidence < 0.25:
        return (
            keyword_category,
            bert_confidence,
            "Hybrid (Keyword Assisted)",
            matched_keywords
        )

    # --------------------------------------------------------
    # Case 5: Otherwise trust IndicBERT
    # --------------------------------------------------------

    return (
        bert_category,
        bert_confidence,
        "IndicBERT",
        []
    )


# ============================================================
# Single Prediction
# ============================================================

def predict_category(complaint):

    if model is None:
        raise RuntimeError(
            f"Model is not loaded: {MODEL_LOAD_ERROR}"
        )

    inputs = tokenizer(
        complaint,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    bert_id = torch.argmax(probabilities).item()
    bert_category = id2label[bert_id]

    (
        final_category,
        confidence,
        method,
        matched_keywords
    ) = _weighted_hybrid_prediction(
        probabilities,
        complaint
    )

    return {
        "category": final_category,
        "confidence": confidence,
        "method": method,
        "matched_keywords": matched_keywords,
        "bert_category": bert_category
    }


# ============================================================
# Batch Prediction
# ============================================================

def predict_batch(
    complaints,
    batch_size=16,
    progress_callback=None
):

    if model is None:
        raise RuntimeError(
            f"Model is not loaded: {MODEL_LOAD_ERROR}"
        )

    results = []
    total = len(complaints)

    for start in range(0, total, batch_size):

        batch = complaints[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        for i, complaint in enumerate(batch):

            (
                final_category,
                confidence,
                method,
                matched_keywords
            ) = _weighted_hybrid_prediction(
                probabilities[i],
                complaint
            )

            bert_id = torch.argmax(
                probabilities[i]
            ).item()

            bert_category = id2label[bert_id]

            results.append({
                "complaint": complaint,
                "category": final_category,
                "confidence": confidence,
                "method": method,
                "matched_keywords": matched_keywords,
                "bert_category": bert_category
            })

        if progress_callback:
            progress_callback(
                min(start + batch_size, total),
                total
            )

    return results