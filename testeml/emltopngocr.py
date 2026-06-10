import os
import email
from email import policy
from email.parser import BytesParser
import pandas as pd
import easyocr
from html2image import Html2Image  # E-postayı resme çeviren kütüphane
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- 1. MODELİ EĞİTME KISMI (BEYİN) ---
def train_model(csv_path ):
    print("Model hafızası yükleniyor (CSV okunuyor)...")
    df = pd.read_csv(csv_path)
    # Konu ve gövdeyi birleştir
    df['text'] = df['subject'].fillna('') + " " + df['clean_body'].fillna('')
    
    # Pipeline kur (Vektörleştirici + Sınıflandırıcı)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', SVC())
    ])
    
    # Eğit
    model.fit(df['text'], df['label'])
    print(" Model hazır! Artık gelen resimleri yorumlayabilir.")
    return model

# --- 2. EML -> HTML HAZIRLIK KISMI ---
def get_content_for_screenshot(eml_path):
    """EML dosyasını okur, HTML varsa onu, yoksa düz metni HTML'e çevirip döndürür."""
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    html_content = ""
    plain_text = ""

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/html":
                try: html_content += part.get_content()
                except: pass
            elif ctype == "text/plain":
                try: plain_text += part.get_content()
                except: pass
    else:
        ctype = msg.get_content_type()
        try:
            if ctype == "text/html": html_content = msg.get_content()
            else: plain_text = msg.get_content()
        except: pass

    # Eğer HTML varsa onu kullan, yoksa düz metni HTML formatına sok (Resim çekebilmek için)
    if html_content.strip():
        return html_content
    elif plain_text.strip():
        # Düz metni basit bir HTML içine koyalım ki resmi çekilsin
        return f"<html><body><pre>{plain_text}</pre></body></html>"
    else:
        return None

# --- 3. ANA ANALİZ SÜRECİ (GÖZ + BEYİN) ---
def analyze_single_eml(file_path, model, ocr_reader):
    print(f"\nDosya İşleniyor: {file_path}")
    
    # A) İçeriği (HTML string) al
    html_data = get_content_for_screenshot(file_path)
    if not html_data:
        print("Dosya boş veya okunamadı.")
        return

    # B) Resmi Çek (Render)
    screenshot_name = "gecici_analiz_resmi.png"
    hti = Html2Image()
    # Logları gizle, çıktıyı temiz tut
    hti.output_path = "." 
    
    print(" E-postanın fotoğrafı çekiliyor...")
    hti.screenshot(html_str=html_data, save_as=screenshot_name, size=(800, 1000))

    # C) OCR ile Oku (Göz)
    print("Resim okunuyor (OCR)...")
    # detail=0 sadece metni verir
    result_list = ocr_reader.readtext(screenshot_name, detail=0, paragraph=True)
    extracted_text = " ".join(result_list)
    
    # D) Tahmin Et (Karar)
    if not extracted_text.strip():
        print("Resimde hiç yazı bulunamadı! (Sadece görsel olabilir)")
        # Boşsa genelde spamdir veya belirsizdir, riskli diyebiliriz.
        final_prediction = "UNKNOWN"
    else:
        print(f"Algılanan Metin: {extracted_text[:60]}...") # İlk 60 karakteri göster
        prediction = model.predict([extracted_text])[0]
        final_prediction = prediction.upper()

    print(f"------------------------------------------------")
    print(f"SONUÇ: {final_prediction}")
    print(f"------------------------------------------------")

    # Temizlik: Geçici resmi sil
    if os.path.exists(screenshot_name):
        os.remove(screenshot_name)

# --- 4. ÇALIŞTIRMA ---
if __name__ == "__main__":
    # A) Önce modeli eğit (CSV ile)
    csv_dosyasi = "supervised_emails_deneme.csv" # Senin dosyan
    if os.path.exists(csv_dosyasi):
        brain = train_model(csv_dosyasi)
    else:
        print("CSV bulunamadı, önce onu ayarla.")
        exit()

    print("OCR motoru baslatiliyor.")
    reader = easyocr.Reader(['en', 'tr'], gpu=True)

    #
    hedef_eml_dosyasi = "🎈Çocuk Ürünlerinde P'ye Varan İndirimi Kaçırma!.eml"
   
    if os.path.exists(hedef_eml_dosyasi):
        analyze_single_eml(hedef_eml_dosyasi, brain, reader)
    else:
        print(f"\n'{hedef_eml_dosyasi}' bulunamadı.")
        print("Lütfen kodun olduğu klasöre test etmek için bir .eml dosyası koy ve adını yukarıya yaz.") 