# 🎙️ NeoScribe

**Seçtiğin bir uygulamanın sesini (Opera, Chrome, Teams, Discord…) yakalayıp internet olmadan, canlı olarak yazıya çeviren Windows uygulaması.**

NeoScribe, tek bir uygulamanın ses çıkışını dinler ve [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ile bilgisayarında yazıya döker. Ders videosu, konferans veya toplantı izlerken konuşma metni pencerede canlı akar ve aynı anda `.txt` dosyasına yazılır. İstersen sesi `.wav` olarak da kaydedebilirsin.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)
![License](https://img.shields.io/badge/License-MIT-yellow)

<!-- Ekran görüntüsünü buraya ekle: docs/screenshot.png dosyasını repoya yükle -->
<!-- ![NeoScribe ekran görüntüsü](docs/screenshot.png) -->

## ✨ Özellikler

- **Uygulamaya özel ses yakalama:** Mikrofon veya tüm sistem sesi yerine yalnızca seçtiğin uygulamanın sesi alınır ([proctap](https://pypi.org/project/proctap/) ile)
- **Tamamen çevrim dışı:** Model indirildikten sonra internet gerekmez, ses bilgisayarından çıkmaz
- **Canlı metin:** Konuşma penceredeki metin kutusunda akar ve saat damgasıyla `.txt` dosyasına eklenir
- **Türkçe ve İngilizce:** Yapay zekâ ve Python konulu teknik terimler için hazır ipucu metniyle (transformer, attention, embedding vb.)
- **Model seçimi:** `small`, `medium`, `large-v3-turbo`, `large-v3`
- **GPU (CUDA) veya CPU:** GPU çalışmazsa otomatik olarak daha hafif ayara geçer
- **Akıllı bölme:** Cümleyi ortasından kesmemek için parçaları konuşma aralarındaki en sessiz noktada böler
- **Ses yükseltme:** Düşük seviyeli sesi kayıpsız şekilde yükseltir
- **Sadece ses kaydı modu:** Metne çevirmeden yalnızca `.wav` kaydı alabilirsin
- **Süreç listesi:** Penceresi açık uygulamaları listeler, tarayıcı ve toplantı uygulamalarını otomatik seçer

## 🖥️ Gereksinimler

- **Windows** 10 (güncel sürüm) veya Windows 11. Uygulama başka işletim sistemlerinde çalışmaz.
- **Python 3.10 veya üzeri** (Python 3.13 ile geliştirildi)
- Hızlı çalışması için **NVIDIA ekran kartı** önerilir, yoksa CPU ile de çalışır
- Model dosyaları için boş disk alanı: `small` yüzlerce MB, `large-v3` birkaç GB

## 📦 Kurulum

```bash
# 1. Projeyi indir
git clone https://github.com/canerergun/neoscribe.git
cd neoscribe

# 2. Sanal ortam oluştur ve etkinleştir
python -m venv .venv
.venv\Scripts\activate

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. (Sadece NVIDIA GPU kullanacaksan) CUDA kütüphanelerini kur
pip install -r requirements-gpu.txt
```

Uygulama, pip ile kurulan `nvidia-cublas-cu12` ve `nvidia-cudnn-cu12` paketlerinin DLL klasörlerini kendisi bulup Windows'a tanıtır. Ayrıca CUDA Toolkit kurman gerekmez, güncel bir NVIDIA sürücüsü yeterlidir.

## ▶️ Kullanım

```bash
python neoscribe.py
```

1. Metne çevirmek istediğin uygulamada (ör. Opera'da bir video) sesi oynat.
2. NeoScribe'te **Uygulama / PID** listesinden o uygulamayı seç. Liste boşsa **Yenile**'ye bas.
3. **Dil**, **Model** ve **Cihaz** seç. Başlangıç için `medium` + `GPU (CUDA)` iyi bir denge sunar.
4. **Kaydet** bölümünden istediğin çıktıları işaretle: Metin (TXT), Ses (WAV) veya ikisi.
5. **Başlat**'a bas. İlk çalıştırmada model indirilir, biraz beklemen gerekir.
6. Bitirince **Durdur**'a bas. WAV kaydı açıksa dosyanın yazılmasını bekle, sonra pencereyi kapat.

### Dosyalar

| Çıktı | Varsayılan konum |
|-------|------------------|
| Metin | Çalıştırdığın klasördeki `konusma_metni.txt` |
| Ses | `kayit_sesi_YYYYMMDD_HHMMSS.wav` (eski kayıtların üzerine yazılmaz) |
| Model dosyaları | Hugging Face önbelleği (`%USERPROFILE%\.cache\huggingface`) |

> **Not:** Metin dosyasına yeni konuşmalar **eklenir**, dosya her başlatmada sıfırlanmaz. Temiz bir dosya için **Temizle**'ye bas veya başka bir dosya adı seç.

## 💡 İpuçları

- Uygulamanın ses seviyesini Windows **Ses Karıştırıcısı**'nda yüksek tut, yakalanan ses ne kadar güçlüyse sonuç o kadar iyi olur.
- Tarayıcılar birden çok süreç açar. Ses gelmiyorsa aynı uygulamanın farklı PID'sini dene.
- Metin gecikiyorsa ("geride kalındı" uyarısı) bir küçük model seç.
- Teknik bir konuyu dinliyorsan dili doğru seç, hazır ipucu metni terimleri tanımaya yardım eder.

## 🛠️ Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| "proctap içinde ses yakalama sınıfı bulunamadı" | `pip install --upgrade proctap` |
| cuDNN / cuBLAS hatası | `pip install -r requirements-gpu.txt` ve NVIDIA sürücünü güncelle. Uygulama gerekirse otomatik olarak daha hafif ayara geçer |
| "Yeni ses verisi gelmedi" | Uygulamada ses gerçekten çalıyor mu kontrol et, farklı PID seç |
| "Sessiz parça atlandı" | Uygulamanın ses seviyesini yükselt |
| Kayıttan sonra `.wav.tmp` dosyası kaldı | Program kayıt bitmeden kapandı, ham ses bu dosyada duruyor |
| Model yüklenemedi | İnternet bağlantını kontrol et (ilk indirme için gerekir), yeterli disk alanı olduğundan emin ol |

## ⚖️ Sorumluluk ve Gizlilik

- İşlem tamamen bilgisayarında yapılır, ses veya metin bir sunucuya gönderilmez.
- Toplantı, ders veya görüşme kaydederken ilgili kişilerin izni ve yasal düzenlemeler senin sorumluluğundadır.

## 🧰 Kullanılan Teknolojiler

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): konuşma tanıma
- [proctap](https://pypi.org/project/proctap/): uygulama bazlı ses yakalama
- [NumPy](https://numpy.org/) ve [psutil](https://github.com/giampaolo/psutil): ses işleme ve süreç listesi
- Tkinter: arayüz

## 🗺️ Yol Haritası

- [ ] Daha fazla dil desteği
- [ ] SRT / VTT altyazı çıktısı
- [ ] Çeviri modu (konuşmayı başka dile çevirme)
- [ ] Koyu tema
- [ ] `.exe` paketi (PyInstaller)

## 🤝 Katkı

Hata bildirimi ve önerileri [Issues](https://github.com/canerergun/neoscribe/issues) bölümünden paylaşabilirsin. Pull request'ler açıktır.

## 📄 Lisans

MIT Lisansı ile dağıtılmaktadır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bak.

## 👤 Geliştirici

**Caner Ergün**: [@canerergun](https://github.com/canerergun)

---

## 🇬🇧 English Summary

**NeoScribe** is a Windows desktop app that captures the audio of a single application (browser, Teams, Discord…) and transcribes it live, fully offline, using faster-whisper. It supports Turkish and English, model selection (small to large-v3), GPU (CUDA) or CPU, smart segmentation at silences, volume normalization, live `.txt` output and optional `.wav` recording.

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # only for NVIDIA GPUs
python neoscribe.py
```

Windows only. Licensed under MIT.
