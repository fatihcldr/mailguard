import os
import re
import pandas as pd
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
from urllib.parse import urlparse #url parse islemi icin 

#kendi ham spam ve phishing maillerinizi klasor yollarinda ki sekilde yapip csv elde edebilirsiniz.
# EML KLASÖRLERİ
PATH_HAM = r"dataset/ham"
PATH_SPAM = r"dataset/spam"
PATH_PHISHING = r"dataset/phishing"
OUTPUT_CSV = "xgboost_data_v1.csv"

#  ÖZNİTELİK ÇIKARMA


def analyze_links(soup, sender_domain):
    """
    Linkleri analiz eder:
    - Kaç tanesi HTTPS? (Güven sinyali)
    - Kaç tanesi gönderen domain ile eşleşmiyor? 
    - IP adresi içeren link var mı?
    """
    links = soup.find_all('a', href=True)
    total_links = len(links)
    
    if total_links == 0:
        return 0, 0, 0, 0 # Link yoksa hepsi 0

    num_https = 0
    num_ip_links = 0
    num_mismatch = 0
    
    ip_pattern = re.compile(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

    for link in links:
        href = link['href'].lower()
        
        # 1. HTTPS Kontrolü (Google, Bankalar hep https kullanır)
        if href.startswith("https"):
            num_https += 1
            
        # 2. IP Adresi Kontrolü
        if ip_pattern.search(href):
            num_ip_links += 1
            
        # 3. Domain Uyuşmazlığı (Sender: google.com -> Link: xyzsite.com)
        # Bu basit bir kontrol, domain parse etmek gerekir
        if sender_domain and sender_domain not in href and "http" in href:
            num_mismatch += 1

    https_ratio = num_https / total_links # Oran ne kadar yüksekse o kadar güvenli
    return total_links, https_ratio, num_ip_links, num_mismatch

def extract_features_from_eml(file_path, label_code):
    features = {
        "beody_ln": 0,
        "html_len": 0,
        "num_links": 0,
        "https_link_ratio": 0.0, # YENİ: Güvenli link oranı
        "num_ip_links": 0,       # YENİ: IP link sayısı
        "link_mismatch": 0,      # YENİ: Domain uyuşmazlığı
        "num_imgs": 0,
        "num_inputs": 0,
        "suspicious_word_count": 0,
        "urgent_subject": 0,     # Konuda aciliyet var mı?
        "label": label_code
    }

    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        # 1. Gönderen Domaini Bul (Tutarlılık kontrolü için)
        sender = str(msg['from'])
        sender_domain = ""
        if "@" in sender:
            try:
                sender_domain = sender.split("@")[-1].strip().replace(">", "").lower()
            except: pass

        # 2. Konu Analizi
        subject = str(msg['subject']).lower() if msg['subject'] else ""
        if "urgent" in subject or "acil" in subject or "!" in subject:
            features["urgent_subject"] = 1

        # 3. İçerik Analizi
        html_content = ""
        text_content = ""

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                try:
                    payload = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                    if ctype == "text/html": html_content += payload
                    elif ctype == "text/plain": text_content += payload
                except: pass
        else:
            try:
                payload = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
                if msg.get_content_type() == "text/html": html_content = payload
                else: text_content = payload
            except: pass

        # HTML Analizi
        if html_content:
            soup = BeautifulSoup(html_content, "lxml")
            features["html_len"] = len(html_content)
            features["num_imgs"] = len(soup.find_all('img'))
            features["num_inputs"] = len(soup.find_all('input'))
            
            # Link Analiz Fonksiyonunu Çağır
            total_links, https_ratio, num_ip, mismatch = analyze_links(soup, sender_domain)
            features["num_links"] = total_links
            features["https_link_ratio"] = https_ratio
            features["num_ip_links"] = num_ip
            features["link_mismatch"] = mismatch
            
            if not text_content: text_content = soup.get_text(separator=" ")

        features["body_len"] = len(text_content)
        
        # Kelime Sayacı (Ama artık tek kriter değil)
        keywords = ["verify", "account", "suspend", "bank", "password", "doğrula", "şifre"]
        text_lower = text_content.lower()
        count = 0
        for word in keywords:
            count += text_lower.count(word)
        features["suspicious_word_count"] = count

    except Exception as e:
        print(f"Hata: {e}")
        return None

    return features

# ÇALIŞTIRMA

def process_folder(folder_path, label_code):
    data_list = []
    if not os.path.exists(folder_path): return data_list
    files = [f for f in os.listdir(folder_path) if f.endswith('.eml')]
    print(f"İşleniyor: {folder_path} ({len(files)} dosya)...")
    for file in files:
        feat = extract_features_from_eml(os.path.join(folder_path, file), label_code)
        if feat: data_list.append(feat)
    return data_list

all_data = []
all_data.extend(process_folder(PATH_HAM, 0))      # HAM
all_data.extend(process_folder(PATH_SPAM, 1))     # SPAM
all_data.extend(process_folder(PATH_PHISHING, 2)) # PHISHING

df = pd.DataFrame(all_data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n Veri kaydedildi: {OUTPUT_CSV}") 
print(df.groupby('label')[['suspicious_word_count', 'https_link_ratio', 'num_inputs']].mean())