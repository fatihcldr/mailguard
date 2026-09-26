import os
import pandas as pd
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import re

#kendi ham spam ve phishing maillerinizi klasor yollarinda ki sekilde yapip csv elde edebilirsiniz.
# 1. Klasör Yolları 
PATH_HAM = r"dataset/ham" 
PATH_SPAM = r"dataset/spam"
PATH_PHISHING = r"dataset/phishing"

OUTPUT_CSV = "egitim_verisi.csv"

def clean_html(html_content):
    """HTML etiketlerini temizler ve sadece metni alır."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ")
        # Fazla boşlukları ve satır atlamaları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except:
        return ""

def extract_eml_data(file_path, label):
    """Bir .eml dosyasını okur ve gerekli sütunları çıkarır."""
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # 1. Subject (Konu)
    subject = str(msg['subject']) if msg['subject'] else ""

    # 2. Sender Domain (Gönderen Domaini) ! Oldukca onemli
    sender = str(msg['from'])
    sender_domain = ""
    if "@" in sender:
        try:
            # "Name <user@domain.com>" formatından domaini çekme 
            sender_domain = sender.split("@")[-1].strip().replace(">", "")
        except:
            sender_domain = "unknown"

    # 3. Body (İçerik) - Hem Text hem HTML kontrolü
    body_content = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition:
                try:
                    payload = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                    if content_type == "text/html":
                        body_content += clean_html(payload) + " "
                    elif content_type == "text/plain":
                        body_content += payload + " "
                except:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
            if msg.get_content_type() == "text/html":
                body_content = clean_html(payload)
            else:
                body_content = payload
        except:
            pass

    # Temizlik 
    clean_body = re.sub(r'\s+', ' ', body_content).strip()

    return {
        "subject": subject, #svm icin en onemli columnlar subject, clean_body ve ana karar verme kisminda sender_domain cok onemli 
        "clean_body": clean_body,
        "sender_domain": sender_domain,
        "send_hour": 12, # EML'den saat çekmek zor olabilir, varsayılan değer atadık veya parse edilebilir. #Zaten send hour cok onemli degil egitim sirasinda.
        "label": label
    }

def process_folder(folder_path, label_name):
    data_list = []
    if not os.path.exists(folder_path):
        print(f"Uyarı: {folder_path} klasörü bulunamadı!")
        return data_list
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.eml')]
    print(f"'{label_name}' için {len(files)} dosya işleniyor...")

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            data = extract_eml_data(file_path, label_name)
            # Sadece içeriği dolu olanları alalım
            if data["clean_body"] or data["subject"]: 
                data_list.append(data)
        except Exception as e:
            print(f"Hata ({file}): {e}")
            
    return data_list

# ANA İŞLEM 
all_data = []

# 1. Ham Mailleri Ekle (Label: 0 veya 'ham')
all_data.extend(process_folder(PATH_HAM, "ham"))

# 2. Spam Mailleri Ekle (Label: 1 veya 'spam')
all_data.extend(process_folder(PATH_SPAM, "spam"))

# 3. Phishing Mailleri Ekle (Label: 2 veya 'phishing')
all_data.extend(process_folder(PATH_PHISHING, "phishing"))

# DataFrame Oluştur ve Kaydet
df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nİşlem tamamlandı! Toplam {len(df)} satır veri '{OUTPUT_CSV}' dosyasına kaydedildi.")
print(df["label"].value_counts())