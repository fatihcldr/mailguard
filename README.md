<div align="center">

# MailGuard — Hybrid Email Classification Engine

**FastAPI + Redis based Ham / Spam / Phishing detection system combining SVM, XGBoost and OCR**
**FastAPI + Redis tabanlı, SVM, XGBoost ve OCR'ı birleştiren Ham / Spam / Phishing tespit sistemi**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](#)
[![Redis](https://img.shields.io/badge/Redis-Queue-DC382D?style=flat-square&logo=redis&logoColor=white)](#)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-00A98F?style=flat-square)](#)

[English](#english) · [Türkçe](#türkçe)

</div>

---

## English

### Overview

MailGuard is a hybrid email security system that classifies uploaded `.eml` files into **HAM**, **SPAM**, or **PHISHING**. It combines a fast whitelist check, a FastAPI request layer, an asynchronous Redis-backed worker, and two complementary machine learning models — a text-based **Linear SVM** and a structure-based **XGBoost** classifier — with an OCR fallback for image-embedded text.

### Architecture

```
User
  │
  ▼
FastAPI API
  │
  ├── Fast sender whitelist check
  │        │
  │        └── Trusted domain → return HAM immediately
  │
  ▼
Redis Queue
  │
  ▼
Worker
  │
  ├── HTML structure analysis
  ├── OCR (for image-embedded text)
  ├── Linear SVM (text)
  ├── XGBoost (structural features)
  │
  ▼
Final verdict
```

The system is split into two layers:
- **API layer** (`api.py`) — receives the `.eml` file, runs a fast whitelist check, and enqueues the task.
- **Worker layer** (`back_worker.py`) — consumes the queue, runs the full analysis pipeline, and produces the final verdict.

Redis decouples the API from the analysis engine so the API never blocks on slow OCR/ML operations, and the pipeline stays asynchronous end to end.

### How It Works

1. **Email ingestion** — the user uploads an `.eml` file. The API extracts sender, subject and HTML body, then creates an analysis task. Emails from trusted domains (e.g. `google.com`, `microsoft.com`, `.gov.tr`, `.edu.tr`) are fast-tracked as HAM.
2. **Redis queue** — anything that needs deeper analysis is pushed to a Redis queue, keeping the API responsive.
3. **HTML analysis** — the worker strips HTML tags, extracts plain text, and scores structural risk signals such as hidden content (`display:none`), form/input elements, and excessive HTML density.
4. **OCR fallback** — some phishing attempts embed text as images. The worker rasterizes the HTML, pre-processes it with OpenCV, and runs Tesseract OCR to recover any hidden text, feeding it back into the analysis.
5. **ML classification** — a **Linear SVM** (TF-IDF, uni/bi-grams, `CalibratedClassifierCV`) evaluates the semantic content, while an **XGBoost** model evaluates structural features (link count, HTTPS ratio, IP-based links, input fields, image count, suspicious/urgency keywords).
6. **Decision layer** — the SVM verdict is trusted by default. However, if the email contains input fields or IP-based links, the system can directly flag it as PHISHING regardless of the SVM's confidence. If the SVM's confidence is low, the XGBoost result is used as a tie-breaker.

### Repository Structure

```
.
├── api.py                    # FastAPI entry point (ingestion + whitelist + queueing)
├── back_worker.py            # Redis worker (HTML analysis, OCR, SVM + XGBoost inference)
├── eml_to_csvler/
│   ├── eml_to_csv.py         # Converts raw .eml files into SVM training features
│   └── xgb_eml_to_csv.py     # Converts raw .eml files into XGBoost training features
├── templates/
│   ├── index.html            # Upload UI
│   └── result.html           # Result UI
├── static/
│   └── styles.css
├── tesseract/                # Trained-data files used for OCR (eng/tur)
├── sample_data.csv           # Anonymized sample dataset (subject, html_body, label)
├── xgboost_data_v1.csv       # Structural features for the XGBoost model (no personal data)
├── requirements.txt
└── README.md
```

> Note: raw sample emails (`testeml/`) and the full, unfiltered training dataset are intentionally **not included** in this repository, as they were derived from real personal inboxes. `sample_data.csv` is provided instead, with the same schema (`subject`, `html_body`, `label`) but no real personal content, so the pipeline can still be explored end to end. See [Dataset](#dataset--reproducing-training-data) below.

### Installation

```bash
pip install -r requirements.txt
```

Make sure a Redis instance is running locally (default: `localhost:8443` — check `api.py` for the exact host/port used).

Tesseract OCR must also be installed on the system, with the `eng` and `tur` trained-data files available (see the `Tesseract` folder for the language files this project was tested with).

### Running

```bash
# Terminal 1 — API
python api.py

# Terminal 2 — Worker
python back_worker.py
```

Then open `http://localhost:8000` (or the port FastAPI reports) in your browser.

### Dataset / Reproducing Training Data

This project was trained on the author's own real inbox data, which is **not published** for privacy reasons. To reproduce the pipeline with your own data:

1. Collect `.eml` files from your own mail client (Gmail, Outlook, etc. all support exporting individual emails as `.eml`).
2. Place them under a local `testeml/` folder (create it — it's git-ignored).
3. Run `eml_to_csvler/eml_to_csv.py` and `eml_to_csvler/xgb_eml_to_csv.py` to generate the SVM and XGBoost training tables from your own emails.
4. Train the models on the resulting CSVs.

### Models

| Model | Purpose | Key features |
|---|---|---|
| Linear SVM | Semantic/text classification | TF-IDF, uni/bi-grams, `CalibratedClassifierCV` |
| XGBoost | Structural classification | Link count, HTTPS ratio, IP-based links, input fields, image count, suspicious keywords, urgency signals |

### Disclaimer

This project was built for educational and research purposes to explore hybrid ML-based phishing detection. It is not a production-grade security product.

---

## Türkçe

### Genel Bakış

MailGuard, yüklenen `.eml` dosyalarını **HAM**, **SPAM** veya **PHISHING** olarak sınıflandıran hibrit bir e-posta güvenlik sistemidir. Hızlı bir güvenilir gönderici (whitelist) kontrolü, FastAPI tabanlı bir istek katmanı, Redis destekli asenkron bir worker ve birbirini tamamlayan iki makine öğrenmesi modelini — metin tabanlı **Linear SVM** ve yapı tabanlı **XGBoost** — görsel içine gizlenmiş metinler için bir OCR mekanizmasıyla birlikte kullanır.

### Mimari

```
Kullanıcı
  │
  ▼
FastAPI API
  │
  ├── Hızlı güvenilir gönderici kontrolü
  │        │
  │        └── Güvenilir domain → doğrudan HAM sonucu dön
  │
  ▼
Redis Kuyruğu
  │
  ▼
Worker
  │
  ├── HTML yapı analizi
  ├── OCR (görsele gömülü metinler için)
  ├── Linear SVM (metin)
  ├── XGBoost (yapısal öznitelikler)
  │
  ▼
Nihai karar
```

Sistem iki katmandan oluşur:
- **API katmanı** (`api.py`) — `.eml` dosyasını alır, hızlı bir whitelist kontrolü yapar ve görevi kuyruğa ekler.
- **Worker katmanı** (`back_worker.py`) — kuyruktaki görevleri işler, tam analiz hattını çalıştırır ve nihai kararı üretir.

Redis, API'yi analiz motorundan ayırarak API'nin yavaş OCR/ML işlemleri sırasında bloklanmasını engeller; böylece hat uçtan uca asenkron kalır.

### Çalışma Mantığı

1. **Mail alımı** — kullanıcı bir `.eml` dosyası yükler. API gönderici, konu ve HTML gövdesini çıkarır, ardından bir analiz görevi oluşturur. Güvenilir domainlerden (`google.com`, `microsoft.com`, `.gov.tr`, `.edu.tr` vb.) gelen mailler doğrudan HAM olarak hızlandırılmış şekilde işaretlenir.
2. **Redis kuyruğu** — daha derin analiz gerektiren mailler Redis kuyruğuna eklenir, böylece API bloklanmadan hızlı yanıt verebilir.
3. **HTML analizi** — worker, HTML etiketlerini temizler, düz metni çıkarır ve gizli içerik (`display:none`), form/input elemanları, aşırı HTML yoğunluğu gibi yapısal risk sinyallerini puanlar.
4. **OCR mekanizması** — bazı phishing girişimleri metni doğrudan görsele gömer. Worker HTML'i görsele dönüştürür, OpenCV ile ön işler ve Tesseract OCR ile gizlenmiş metni okuyup analiz sürecine dahil eder.
5. **ML sınıflandırması** — **Linear SVM** (TF-IDF, unigram/bigram, `CalibratedClassifierCV`) anlamsal içeriği değerlendirirken, **XGBoost** modeli yapısal öznitelikleri (link sayısı, HTTPS oranı, IP adresi içeren linkler, input alanları, görsel sayısı, şüpheli/aciliyet kelimeleri) değerlendirir.
6. **Karar mekanizması** — varsayılan olarak SVM sonucuna güvenilir. Ancak mailde input alanı veya IP adresi içeren bağlantı varsa, sistem SVM güveninden bağımsız olarak doğrudan PHISHING kararı verebilir. SVM düşük güven ürettiğinde ise XGBoost sonucu belirleyici olarak devreye girer.

### Klasör Yapısı

```
.
├── api.py                    # FastAPI giriş noktası (alım + whitelist + kuyruklama)
├── back_worker.py            # Redis worker (HTML analizi, OCR, SVM + XGBoost çıkarımı)
├── eml_to_csvler/
│   ├── eml_to_csv.py         # Ham .eml dosyalarını SVM eğitim özniteliklerine çevirir
│   └── xgb_eml_to_csv.py     # Ham .eml dosyalarını XGBoost eğitim özniteliklerine çevirir
├── templates/
│   ├── index.html            # Yükleme arayüzü
│   └── result.html            # Sonuç arayüzü
├── static/
│   └── styles.css
├── tesseract/                # OCR icin egitilmis dil dosyalari (eng/tur)
├── sample_data.csv           # Anonimlestirilmis ornek veri seti (subject, html_body, label)
├── xgboost_data_v1.csv       # XGBoost modeli icin yapisal oznitelikler (kisisel veri icermez)
├── requirements.txt
└── README.md
```

> Not: gerçek kişisel gelen kutularından türetildiği için ham örnek e-postalar (`testeml/`) ve tam/filtrelenmemiş eğitim veri seti bu depoya **dahil edilmemiştir**. Bunun yerine, aynı şemaya (`subject`, `html_body`, `label`) sahip ama gerçek kişisel içerik barındırmayan `sample_data.csv` sağlanmıştır; böylece hat uçtan uca incelenebilir. Aşağıdaki [Veri Seti](#veri-seti--kendi-eğitim-verinizi-oluşturma) bölümüne bakın.

### Kurulum

```bash
pip install -r requirements.txt
```

Yerelde bir Redis örneğinin çalışıyor olduğundan emin olun (varsayılan: `localhost:8443` — tam host/port için `api.py` dosyasına bakın).

Sistemde Tesseract OCR'ın kurulu olması ve `eng` ile `tur` dil dosyalarının erişilebilir olması gerekir (projenin test edildiği dil dosyaları için `Tesseract` klasörüne bakabilirsiniz).

### Çalıştırma

```bash
# Terminal 1 — API
python api.py

# Terminal 2 — Worker
python back_worker.py
```

Ardından tarayıcıda `http://localhost:8000` adresini açın (veya FastAPI'nin bildirdiği portu kullanın).

### Veri Seti / Kendi Eğitim Verinizi Oluşturma

Bu proje, yazarın kendi gerçek gelen kutusu verileriyle eğitilmiştir ve gizlilik nedeniyle **paylaşılmamaktadır**. Hattı kendi verinizle yeniden üretmek için:

1. Kendi e-posta istemcinizden (Gmail, Outlook vb.) `.eml` dosyaları dışa aktarın.
2. Bunları yerel bir `testeml/` klasörüne yerleştirin (bu klasörü kendiniz oluşturun — git tarafından yok sayılır).
3. `eml_to_csvler/eml_to_csv.py` ve `eml_to_csvler/xgb_eml_to_csv.py` dosyalarını çalıştırarak kendi mailinizden SVM ve XGBoost eğitim tablolarını üretin.
4. Modelleri elde ettiğiniz CSV'ler üzerinde eğitin.

### Modeller

| Model | Amaç | Temel özellikler |
|---|---|---|
| Linear SVM | Anlamsal/metin sınıflandırması | TF-IDF, unigram/bigram, `CalibratedClassifierCV` |
| XGBoost | Yapısal sınıflandırma | Link sayısı, HTTPS oranı, IP adresi içeren linkler, input alanları, görsel sayısı, şüpheli kelimeler, aciliyet sinyalleri |

### Sorumluluk Reddi

Bu proje, hibrit ML tabanlı phishing tespitini keşfetmek amacıyla eğitim ve araştırma amaçlı geliştirilmiştir; production seviyesinde bir güvenlik ürünü değildir.
