import os
import re
import uuid
import subprocess
import cv2
import time
import pytesseract
import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
# Makine Öğrenmesi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# =====================================================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
WKHTMLTOIMAGE_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe"
TEST_FOLDER = "testeml"
CSV_TEXT_PATH = "yeni_egitim_verisi.csv"
CSV_META_PATH = "xgboost_data_v2.csv"


# Varsayılan dil (Eğer tespit edilemezse bu ikiliye düşecek)
DEFAULT_OCR_LANG = 'turfast+engfast'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
LABEL_MAP = {0: "HAM", 1: "SPAM", 2: "PHISHING"}

# =====================================================
# 🛡️ GELİŞMİŞ WHITELIST & TYPOSQUATTING
# =====================================================
def is_trusted_sender(sender_email):
    sender = sender_email.lower()
    domain = ""
    if "@" in sender:
        domain = sender.split("@")[-1].strip().replace(">", "")
    
    safe_extensions = [".edu.tr", ".edu", ".gov.tr", ".gov", ".mil", ".k12.tr", ".bel.tr"]
    if any(domain.endswith(ext) for ext in safe_extensions):
        return "TRUSTED", f"Resmi Kurum ({domain})"

    trusted_companies = [
        "google.com", "gmail.com", "microsoft.com", "office.com", 
        "amazon.com", "linkedin.com","instagram.com","facebook.com", "apple.com",
        "turkiye.gov.tr", "uyap.gov.tr", "akbank.com", "garanti.com.tr",
        "isbank.com.tr", "yapikredi.com.tr" 
    ]
    
    if domain in trusted_companies:
        return "TRUSTED", f"Kurumsal Domain ({domain})"

    for safe_domain in trusted_companies:
        safe_name = safe_domain.split(".")[0]
        check_name = domain.split(".")[0]
        similarity = SequenceMatcher(None, safe_name, check_name).ratio()
        
        if similarity > 0.85 and safe_name != check_name:
            return "PHISHING_TRAP", f"Taklit Domain Tespiti! ({domain} ~ {safe_domain})"

    return "UNKNOWN", None

# =====================================================
# [MODEL EĞİTİMİ] (RAM Cache Zaten Burada Yapılıyor)
# =====================================================
def train_text_model(csv_path):
    print("[INFO] Metin Modeli (SVM) yukleniyor...")
    try:
        df = pd.read_csv(csv_path).dropna(subset=['clean_body', 'label'])
        X = df['subject'].fillna('') + " " + df['clean_body']
        y = df['label']
        svm = LinearSVC(dual="auto")
        clf = CalibratedClassifierCV(svm) 
        model = Pipeline([('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))), ('clf', clf)])
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"[ERROR] SVM Hatasi: {e}")
        return None

def train_structure_model(csv_path):
    print("[INFO] Yapisal Model (XGBoost) yukleniyor...")
    try:
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['label'])
        y = df['label']
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='mlogloss')
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"[ERROR] XGBoost Hatasi: {e}")
        return None

# =====================================================
# [FEATURE EXTRACTOR]
# =====================================================
def analyze_intent(text):
    text = text.lower()
    urgency_patterns = [r"immediately", r"urgen(t|cy)", r"as soon as possible", r"within \d+ hours", r"suspend", r"terminate", r"close.*account", r"derhal", r"hemen", r"acil", r"son uyarı", r"kapanacak"]
    financial_patterns = [r"invoice", r"payment", r"cost", r"price", r"bitcoin", r"bank", r"transfer", r"fatura", r"ödeme", r"tutar", r"borç", r"hesap no"]
    authority_patterns = [r"police", r"law", r"legal action", r"court", r"arrest", r"security alert", r"unauthorized", r"polis", r"yasal işlem", r"mahkeme", r"avukat", r"yetkisiz"]
    cta_patterns = [r"click here", r"login", r"sign in", r"verify", r"update", r"download", r"tıkla", r"giriş yap", r"doğrula", r"güncelle", r"indir"]
    
    return {
        "intent_urgency": sum(1 for p in urgency_patterns if re.search(p, text)),
        "intent_financial": sum(1 for p in financial_patterns if re.search(p, text)),
        "intent_authority": sum(1 for p in authority_patterns if re.search(p, text)),
        "intent_cta": sum(1 for p in cta_patterns if re.search(p, text))
    }

def extract_features_live(html_content, text_content, subject, sender):
    features = {
        "body_len": len(text_content), "html_len": len(html_content), "num_links": 0, "https_link_ratio": 0.0,
        "num_ip_links": 0, "link_mismatch": 0, "num_imgs": 0, "num_inputs": 0, "suspicious_word_count": 0, "urgent_subject": 0
    }
    
    if "urgent" in subject.lower() or "acil" in subject.lower() or "!" in subject: features["urgent_subject"] = 1

    if html_content:
        soup = BeautifulSoup(html_content, "lxml")
        features["num_imgs"] = len(soup.find_all('img'))
        features["num_inputs"] = len(soup.find_all('input'))
        links = soup.find_all('a', href=True)
        features["num_links"] = len(links)
        
        if len(links) > 0:
            sender_domain = sender.split("@")[-1].strip().replace(">", "").lower() if "@" in sender else ""
            num_https, num_ip, mismatch = 0, 0, 0
            ip_pattern = re.compile(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
            for link in links:
                href = link['href'].lower()
                if href.startswith("https"): num_https += 1
                if ip_pattern.search(href): num_ip += 1
                if sender_domain and sender_domain not in href and "http" in href: mismatch += 1
            
            features["https_link_ratio"] = num_https / len(links)
            features["num_ip_links"] = num_ip
            features["link_mismatch"] = mismatch

    keywords = ["verify", "account", "suspend", "bank", "password", "doğrula", "şifre"]
    features["suspicious_word_count"] = sum(text_content.lower().count(w) for w in keywords)
    
    intents = analyze_intent(text_content)
    features.update(intents)

    return pd.DataFrame([features])

# =====================================================
# [OCR MOTORU - OPTİMİZE EDİLMİŞ]
# =====================================================
class EmailOCRProcessor:
    def __init__(self):
        # Hız için tessedit_do_invert=0 ve psm 6
        self.custom_config = r'--oem 1 --psm 6 -c tessedit_do_invert=0'
        self.default_lang = DEFAULT_OCR_LANG

    def clean_html_pro(self, html_content):
        try:
            soup = BeautifulSoup(html_content, "lxml")
            for tag in soup(["script", "style", "head", "title", "meta", "noscript"]): tag.decompose()
            return re.sub(r'\s+', ' ', soup.get_text(separator=" ")).strip()
        except: return ""

    def calculate_risk_score(self, html_content):
        risk_score = 0
        reasons = []
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text().strip()
        
        if len(html_content) > 500 and len(text_content) < (len(html_content) * 0.1):
            risk_score += 50; reasons.append("Resim Agirlikli")
        if soup.find_all(style=re.compile(r'(font-size:\s*0|display:\s*none)', re.IGNORECASE)):
            risk_score += 30; reasons.append("Gizli Metin")
        if soup.find_all('input'):
            risk_score += 20; reasons.append("Input Var")
        
        suspicious = ['microsoft', 'office', 'secure', 'bank', 'fatura', 'verify']
        for img in soup.find_all('img'):
            if any(k in img.get('src', '').lower() for k in suspicious):
                risk_score += 40; reasons.append("Supheli Logo"); break
        
        return risk_score, reasons

    #  YENİ: Dışarıdan dil parametresi alıyor
    def run_ocr_from_image_path(self, image_path, specific_lang=None):
        try:
            img = cv2.imread(image_path)
            if img is None: return ""
            
            # Grayscale + Threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Binary threshold metni keskinleştirir ama bazen bozar, duruma göre
            _, bw = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            processed = cv2.resize(bw, None, fx=1.5, fy=1.5)
            
            # HANGİ DİL?
            lang_to_use = specific_lang if specific_lang else self.default_lang
            
            data = pytesseract.image_to_data(processed, lang=lang_to_use, config=self.custom_config, output_type=pytesseract.Output.DICT)
            tokens = [data['text'][i].strip() for i in range(len(data['text'])) if int(data['conf'][i]) > 60 and len(data['text'][i]) > 2]
            return " ".join(tokens)
        except Exception as e:
            print(f"OCR Hatasi: {e}"); return ""

def convert_html_to_image(html_content, output_path):
    temp_html = f"temp_{uuid.uuid4().hex}.html"
    with open(temp_html, "w", encoding="utf-8") as f: f.write(html_content)
    try:
        subprocess.run([WKHTMLTOIMAGE_PATH, "--quality", "50", "--width", "800", "--quiet", temp_html, output_path], check=True)
        os.remove(temp_html)
        return True
    except:
        if os.path.exists(temp_html): os.remove(temp_html)
        return False

# =====================================================
# [MAIN - KONTROL MERKEZİ]
# =====================================================
def main():
    print("==========================================")
    print("   SIBER GUVENLIK MAIL MOTORU v3.2 (Hızlı)")
    print("==========================================")
    
    #  Modeller burada yüklenip RAM'e alınıyor.
    # Döngü dışında olduğu için "Persistent" gibi davranır.
    text_model = train_text_model(CSV_TEXT_PATH)
    struct_model = train_structure_model(CSV_META_PATH)
    ocr_engine = EmailOCRProcessor()
    
    if not os.path.exists(TEST_FOLDER): os.makedirs(TEST_FOLDER); return
    files = [f for f in os.listdir(TEST_FOLDER) if f.endswith('.eml')]
    
    print(f"\n[INFO] {len(files)} adet mail test ediliyor...\n")

    for file_name in files:
        print(f">>> ANALİZ: {file_name}")
        start_time = time.time()
        
        try:
            with open(os.path.join(TEST_FOLDER, file_name), 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
        except:
            print("   [ERROR] Dosya okunamadi."); continue
            
        subject = str(msg['subject']) if msg['subject'] else ""
        sender = str(msg['from'])
        
        # --- 1. WHITELIST ---
        trust_status, reason = is_trusted_sender(sender)
        if trust_status == "TRUSTED":
            print(f"   [INFO] Otomatik Guvenli: {reason}")
            print(f"   [SONUC] FINAL KARAR: HAM")
            print("-" * 50); continue
        elif trust_status == "PHISHING_TRAP":
            print(f"   [ALERT] KRITIK UYARI: {reason}")
            print(f"   [SONUC] FINAL KARAR: PHISHING"); 
            print("-" * 50); continue

        html_content = ""
        if msg.get_body(preferencelist=('html')): html_content = msg.get_body(preferencelist=('html')).get_content()
        elif msg.get_body(preferencelist=('plain')): html_content = msg.get_body(preferencelist=('plain')).get_content()
        
        clean_text_svm = ocr_engine.clean_html_pro(html_content)
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text(separator=" ")

        # ---  2. LANG DETECT & OCR ---
        risk_score, reasons = ocr_engine.calculate_risk_score(html_content)
        
        if (risk_score >= 40) or (len(clean_text_svm) < 50):
            print(f"   [RISK] Puan: {risk_score} -> OCR Gerekiyor...")
            
            # A) DİLİ TESPİT ET (Speed Hack)
            detected_ocr_lang = None
            try:
                # Sadece yeterli metin varsa tespit et
                if len(clean_text_svm) > 15:
                    lang_code = detect(clean_text_svm)
                    if lang_code == 'tr': 
                        detected_ocr_lang = 'turfast'
                        print("   [AKILLI OCR] Dil: Türkçe tespit edildi. (Hızlı Mod)")
                    elif lang_code == 'en': 
                        detected_ocr_lang = 'engfast'
                        print("   [AKILLI OCR] Dil: İngilizce tespit edildi. (Hızlı Mod)")
            except:
                pass # Hata olursa None kalır, default (tur+eng) çalışır

            # B) GÖRÜNTÜ OLUŞTUR VE OCR ÇALIŞTIR
            temp_img = "temp_scan.jpg"
            if convert_html_to_image(html_content, temp_img):
                # Tespit edilen dili parametre olarak gönderiyoruz
                ocr_text = ocr_engine.run_ocr_from_image_path(temp_img, specific_lang=detected_ocr_lang)
                clean_text_svm += " " + ocr_text
                if os.path.exists(temp_img): os.remove(temp_img)
        
        # --- 3. MODEL TAHMİNLERİ ---
        text_pred, text_prob = "BILINMIYOR", 0.0
        if text_model:
            probs = text_model.predict_proba([subject + " " + clean_text_svm])[0]
            max_index = np.argmax(probs)
            text_pred = text_model.classes_[max_index].upper()
            text_prob = probs[max_index]

        struct_pred = "BILINMIYOR"
        current_features = None
        if struct_model:
            features_df = extract_features_live(html_content, text_content, subject, sender)
            current_features = features_df.iloc[0]
            struct_res = struct_model.predict(features_df)[0]
            struct_pred = LABEL_MAP.get(struct_res, "BILINMIYOR")

        # --- 4. KARAR MEKANİZMASI ---
        final_verdict = "HAM"
        decision_reason = "Normal Analiz"
        is_hard_phishing = False
        
        if current_features is not None:
            if current_features['num_inputs'] > 0:
                final_verdict = "PHISHING"; decision_reason = "Input Formu Var"; is_hard_phishing = True
            elif current_features['num_ip_links'] > 0:
                final_verdict = "PHISHING"; decision_reason = "IP Linki Var"; is_hard_phishing = True
        
        if not is_hard_phishing:
            if struct_pred == "PHISHING" and text_pred == "HAM" and text_prob > 0.50:
                final_verdict = "HAM"; decision_reason = f"SVM Güveni (%{int(text_prob*100)})"
            elif struct_pred == "SPAM" and text_pred == "HAM" and text_prob > 0.50:
                final_verdict = "HAM"; decision_reason = "Reklam Yapısı Gözardı"
            elif text_pred == struct_pred:
                final_verdict = text_pred; decision_reason = f"Ortak Karar ({text_pred})"
            else:
                if text_pred == "PHISHING" or struct_pred == "PHISHING":
                    final_verdict = "PHISHING"; decision_reason = "Güvenlik Önceliği"
                elif text_pred == "SPAM" or struct_pred == "SPAM":
                    final_verdict = "SPAM"; decision_reason = "Spam Şüphesi"

        elapsed_time = time.time() - start_time
        print(f"   [METIN] SVM: {text_pred} (%{int(text_prob*100)}) | [YAPI] XGB: {struct_pred}")
        print(f"   [DETAY] {decision_reason}")
        print(f"   [SONUC] {final_verdict} ({elapsed_time:.3f}s)")
        print("-" * 50)

if __name__ == "__main__": main()

class MailDetectorEngine:
    def __init__(self):
        self.text_model = train_text_model(CSV_TEXT_PATH)
        self.struct_model = train_structure_model(CSV_META_PATH)
        self.ocr_engine = EmailOCRProcessor()

    def analyze_eml_bytes(self, eml_bytes: bytes):
        start_time = time.time()
        msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)

        subject = str(msg['subject']) if msg['subject'] else ""
        sender = str(msg['from'])

        trust_status, reason = is_trusted_sender(sender)
        if trust_status == "TRUSTED":
            return {
                "final": "HAM",
                "svm": "HAM",
                "xgb": "HAM",
                "confidence": 1.0,
                "detail": reason,
                "time": 0.0
            }

        if trust_status == "PHISHING_TRAP":
            return {
                "final": "PHISHING",
                "svm": "PHISHING",
                "xgb": "PHISHING",
                "confidence": 1.0,
                "detail": reason,
                "time": 0.0
            }

        html_content = ""
        if msg.get_body(preferencelist=('html')):
            html_content = msg.get_body(preferencelist=('html')).get_content()
        elif msg.get_body(preferencelist=('plain')):
            html_content = msg.get_body(preferencelist=('plain')).get_content()

        clean_text = self.ocr_engine.clean_html_pro(html_content)
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text(separator=" ")

        risk_score, _ = self.ocr_engine.calculate_risk_score(html_content)

        if risk_score >= 40 or len(clean_text) < 50:
            temp_img = "temp_fastapi.jpg"
            if convert_html_to_image(html_content, temp_img):
                ocr_text = self.ocr_engine.run_ocr_from_image_path(temp_img)
                clean_text += " " + ocr_text
                os.remove(temp_img)

        probs = self.text_model.predict_proba([subject + " " + clean_text])[0]
        svm_idx = np.argmax(probs)
        svm_pred = self.text_model.classes_[svm_idx].upper()
        svm_prob = float(probs[svm_idx])

        feats = extract_features_live(html_content, text_content, subject, sender)
        xgb_res = self.struct_model.predict(feats)[0]
        xgb_pred = LABEL_MAP.get(xgb_res, "BILINMIYOR")

        final = svm_pred if svm_pred == xgb_pred else (
            "PHISHING" if "PHISHING" in (svm_pred, xgb_pred)
            else "SPAM" if "SPAM" in (svm_pred, xgb_pred)
            else "HAM"
        )

        elapsed = time.time() - start_time

        return {
            "final": final,
            "svm": svm_pred,
            "xgb": xgb_pred,
            "confidence": round(svm_prob, 3),
            "detail": "Terminal karar mantığı",
            "time": round(elapsed, 3)
        }
