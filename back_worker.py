import redis #kafka yerine redis cunku kuyrukta onlarca yuzlerce mail beklemiyor tek bir mail bekliyor
import json
import time #Sure hesaplamasi icin gerekli 
import os
import subprocess
import cv2 #Ocr'in daha basarili ve hizli calismasi icin gerekli onislemleri yapmami sagladi 
import re
import pandas as pd
import numpy as np
import pytesseract
from langdetect import detect, LangDetectException #ocrdan langdetect yapmak yerine direkt langdetect kutuphanesinden almak modeli hizlandirdi
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV #Svmin basarisina guvenini hesaplamak icin gerekli 
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier #karar agaclari ile karmasikliklari yakalar random foresta gore genellemesi daha iyi 

# --- AYARLAR ---
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #tesseractin yolu
WKHTMLTOIMAGE_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe" #wkhtmltoimage yolu html2imageden daha hizli ve basarili overheadi azalttim
CSV_TEXT_PATH = "sample_data.csv" #svm icin veri seti
CSV_META_PATH = "xgboost_data_v1.csv"#ayni verilerle fakat farkli iceriklere sahip tree tabanli ml algoritmalari (benim modelimde xgboost) icin verisetim.
OCR_LANG = 'turfast+engfast'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
LABEL_MAP = {0: "HAM", 1: "SPAM", 2: "PHISHING"}

r = redis.Redis(host='localhost', port=8443, db=0) #redise baglanma

# 1. MODELLER

def train_text_model(csv_path):
    print("Metin Modeli (SVM) yükleniyor...")
    try:
        df = pd.read_csv(csv_path).dropna(subset=['clean_body', 'label'])
        X = df['subject'].fillna('') + " " + df['clean_body'] #X subject ve clean_bodye esit oldu
        y = df['label']
        model = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), ('clf', CalibratedClassifierCV(LinearSVC(dual="auto")))])
        model.fit(X, y)
        return model
    except Exception as e:
        print(f" SVM Hatası: {e}")
        return None

def train_structure_model(csv_path):
    print(" Yapısal Model (XGBoost) yükleniyor...")
    try:
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['label'])
        y = df['label']
        model = XGBClassifier(n_estimators=50, max_depth=4, eval_metric='mlogloss')
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"XGBoost Hatası: {e}")
        return None

# 2. OPTİMİZE EDİLMİŞ GÖRÜNTÜ İŞLEME (RAM-ONLY)

class EmailOCRProcessor:
    def __init__(self):
        self.custom_config = r'--oem 1 --psm 6 -c tessedit_do_invert=0'
        self.lang = OCR_LANG

    def clean_html_pro(self, html_content):
        try:
            soup = BeautifulSoup(html_content, "lxml") #daha hizli calismasi icin lxml kullandik
            for tag in soup(["script", "style", "head", "title", "meta"]): tag.decompose() #bu tur html ifadelerini beatifulsoup ile temizlettik
            return re.sub(r'\s+', ' ', soup.get_text(separator=" ")).strip()
        except: return ""

    def calculate_risk_score(self, html_content):
        risk_score = 0
        if not html_content: return 0
        
        if len(html_content) > 500:
            text_len = len(re.sub(r'<[^>]+>', '', html_content))
            if text_len < (len(html_content) * 0.1): risk_score += 50 #eger html contenti 500 den uzunsa risk skoru artar
            
        if "display:none" in html_content or "display: none" in html_content: risk_score += 30 #eger display de hicbirsey yoksa gizlenmis phishing olabilir risk skoru yine artar
        if "<input" in html_content: risk_score += 20 #eger input varsa genelde phishing risk almamak icin risk skoru artti
        return risk_score

    def run_ocr_on_memory(self, img_array, detected_lang=None):
        try:
            if img_array is None: return ""
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) #ocrin hizli taraybilmesi icin Rgb yapiyi griye cevir
            
            use_lang = self.lang
            if detected_lang == 'tr': use_lang = 'turfast'
            elif detected_lang == 'en': use_lang = 'engfast'

            text = pytesseract.image_to_string(gray, lang=use_lang, config=self.custom_config)
            return " ".join(text.split())
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

def convert_html_to_image_ram(html_content):
    try:
        cmd = [
            WKHTMLTOIMAGE_PATH, #bu degerleri burada sabitledim qualityi ve width degerini dusurdum (hizdan kazanc icin)
            "--quality", "30", 
            "--width", "600", 
            "--disable-smart-width",
            "--quiet", 
            "-", "-"
        ]
        #ocri ramde bellekde hizli ve basitce calistirir.
        process = subprocess.run(
            cmd,
            input=html_content.encode('utf-8'),
            capture_output=True,
            check=True
        )
        nparr = np.frombuffer(process.stdout, np.uint8) #ram tabanli goruntu isleme icin stdout
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except:
        return None


# 3. YARDIMCI ANALİZLER

def analyze_intent(text):
    text = text.lower()
    # Bu tur kelime iceren mailleri model icinde siniflandir
    urgency = sum(1 for p in [r"immediately", r"urgen", r"hemen", r"acil", r"kapanacak", r"suspend"] if re.search(p, text))
    financial = sum(1 for p in [r"invoice", r"payment", r"bank", r"fatura", r"ödeme", r"borç", r"cost"] if re.search(p, text))
    authority = sum(1 for p in [r"police", r"legal", r"court", r"polis", r"yasal", r"avukat"] if re.search(p, text))
    cta = sum(1 for p in [r"click", r"login", r"sign in", r"tıkla", r"giriş", r"indir"] if re.search(p, text))
    
    return {
        "intent_urgency": urgency, #xgb icin kritik
        "intent_financial": financial,#xgb icin kritik
        "intent_authority": authority,
        "intent_cta": cta
    }

def extract_features_live(html_content, text_content, subject, sender):
    features = {
        "body_len": len(text_content), "html_len": len(html_content), "num_links": 0, "https_link_ratio": 0.0,
        "num_ip_links": 0, "link_mismatch": 0, "num_imgs": 0, "num_inputs": 0, "suspicious_word_count": 0, "urgent_subject": 0
    }
    
    if "urgent" in subject.lower() or "acil" in subject.lower() or "!" in subject: 
        features["urgent_subject"] = 1
    
    if html_content:
        soup = BeautifulSoup(html_content, "lxml")
        features["num_imgs"] = len(soup.find_all('img'))
        features["num_inputs"] = len(soup.find_all('input'))#xgb icin kritik
        links = soup.find_all('a', href=True)
        features["num_links"] = len(links)
        #xgboost icin gerekli bazi islemler
        if len(links) > 0:
            sender_domain = sender.split("@")[-1].strip().replace(">", "").lower() if "@" in sender else ""
            num_https = 0
            num_ip = 0
            mismatch = 0
            ip_pattern = re.compile(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
            
            for link in links:
                href = link['href'].lower()
                if href.startswith("https"): num_https += 1
                if ip_pattern.search(href): num_ip += 1
                if sender_domain and sender_domain not in href and "http" in href: mismatch += 1

            features["https_link_ratio"] = num_https / len(links)
            features["num_ip_links"] = num_ip #xgb icin kritik
            features["link_mismatch"] = mismatch #xgb icin kritik

    keywords = ["verify", "account", "suspend", "bank", "password", "doğrula", "şifre"]
    features["suspicious_word_count"] = sum(text_content.lower().count(w) for w in keywords)
    intents = analyze_intent(text_content)
    features.update(intents)
    # -------------------------------------------

    return pd.DataFrame([features])

# =====================================================
# 4. WORKER LOOP
# =====================================================
def main_worker():
    print("==========================================")
    print("   Mail Tespit Sistemi    ")
    print("==========================================")
    
    text_model = train_text_model(CSV_TEXT_PATH)
    struct_model = train_structure_model(CSV_META_PATH)
    ocr_engine = EmailOCRProcessor()
    
    print("Worker Hazır Kuyruk (Queue) bekleniyor.")
    
    while True:
        try:
            packed = r.blpop('email_queue', timeout=5)
            if not packed:
                continue

            _, task_data_json = packed
            task = json.loads(task_data_json)

            job_id = task.get('job_id')
            html_content = task.get('html_content', "")
            subject = task.get('subject', "")
            sender = task.get('sender', "")
            file_name = task.get('file', "")  
            
            print(f" [Job: {job_id}] İşleniyor...")
            start_t = time.time()
            
            # 1. Hızlı Text Temizliği
            clean_text_svm = ocr_engine.clean_html_pro(html_content)
            detected_lang = None
            if len(clean_text_svm) > 20:
                try: detected_lang = detect(clean_text_svm) #htmlden textin cikarilip langdetect ile textin dil tespiti 
                except: pass

            # 2. Risk Analizi ve RAM-OCR
            ocr_text_output = ""
            risk_score = ocr_engine.calculate_risk_score(html_content)
            
            if (risk_score >= 40) or (len(clean_text_svm) < 50):
                img_in_memory = convert_html_to_image_ram(html_content)
                if img_in_memory is not None:
                    ocr_text = ocr_engine.run_ocr_on_memory(img_in_memory, detected_lang=detected_lang)
                    if ocr_text:
                        clean_text_svm += " " + ocr_text
                        ocr_text_output = ocr_text

            # 3. Tahminler
            text_pred, text_prob = "BİLİNMİYOR", 0.0
            if text_model:
                probs = text_model.predict_proba([subject + " " + clean_text_svm])[0]
                idx = np.argmax(probs)
                text_pred = text_model.classes_[idx].upper()
                text_prob = probs[idx]

            struct_pred = "BİLİNMİYOR"
            current_features = None
            if struct_model:
                soup_text = BeautifulSoup(html_content, "lxml").get_text(separator=" ")
                feats = extract_features_live(html_content, soup_text, subject, sender)
                current_features = feats.iloc[0]
                res = struct_model.predict(feats)[0]
                struct_pred = LABEL_MAP.get(res, "BİLİNMİYOR")

            # 4. Karar Mantığı (SVM Otoritesinde)
            final_verdict = "HAM"
            reason = "Normal Analiz"
            is_hard = False
            
            if current_features is not None:
                if current_features['num_inputs'] > 0:
                    final_verdict = "PHISHING"; reason = "Kritik: Input Formu Var"; is_hard = True #input varsa phishing
                elif current_features['num_ip_links'] > 0:
                    final_verdict = "PHISHING"; reason = "Kritik: IP Linki Var"; is_hard = True #ip link varsa phishing 
            
            if not is_hard:
                if text_prob >= 0.50: #svm %50'nin altinda dogruluga dusmedigi surece ana karar icin  svme guven (yaptigim testler sonucu boyle uyguladim)
                    final_verdict = text_pred
                    reason = f"SVM Baskın (%{int(text_prob*100)})"
                    if text_pred != struct_pred: reason += f" - (XGB: {struct_pred} yoksayıldı)"
                else:
                    final_verdict = struct_pred
                    reason = f"SVM Kararsız > XGBoost Kararı ({struct_pred})"
            #sonuc kayit
            process_time = time.time() - start_t
            result = {
                "status": "COMPLETED",
                "final_verdict": final_verdict,
                "reason": reason,
                "process_time": process_time,
                "time_ms": process_time * 1000,

                "file": file_name,  #file adi frontta yazsin diye

                "ocr_details": {"full_text": ocr_text_output} if ocr_text_output else None,
                "svm_result": f"{text_pred} (%{int(text_prob*100)})",
                "xgb_result": struct_pred,
            }

            r.setex(f"result:{job_id}", 600, json.dumps(result)) #sonucun apida ve  okunmasi icin gereklidir 

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    main_worker()