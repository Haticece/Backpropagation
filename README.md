## Proje Adı: Backpropagation Algoritmasının Sıfırdan Uygulanması

# Açıklama:

Bu proje, çok katmanlı yapay sinir ağlarında kullanılan Backpropagation (Geriye Yayılım) algoritmasının Python ve NumPy kütüphaneleri kullanılarak sıfırdan (kütüphanelerin hazır fonksiyonları kullanılmadan) nasıl uygulanacağını adım adım göstermektedir. Algoritmanın altında yatan matematiksel çözüm ise ayrı bir dosyada detaylı olarak açıklanmıştır.

# İçerik:

Jupyter Notebook dosyasında (Backpropagation.ipynb) aşağıdaki konular detaylı olarak ele alınmaktadır:

# Yapay Sinir Ağları Temelleri:
Yapay sinir ağlarının yapısı ve çalışma prensibi.
Nöronlar, katmanlar, ağırlıklar ve bias kavramları.
İleri besleme (Forward Propagation) süreci.

# Aktivasyon Fonksiyonları:
Sigmoid aktivasyon fonksiyonu ve türevi.

# Backpropagation Algoritması:
Hata fonksiyonu (Loss Function) ve gradyan hesaplama.
Zincir kuralı (Chain Rule) kullanımı.
Ağırlıkların ve biasların güncellenmesi.
Öğrenme oranı (Learning Rate) kavramı.

# Uygulama:
İki gizli katmanlı bir sinir ağı modeli oluşturma.
Modeli eğitim verileri üzerinde eğitme.
Modelin performansını değerlendirme.
Farklı öğrenme oranlarının etkisini gözlemleme.

# Görselleştirmeler:
Eğitim süreci boyunca hata fonksiyonunun (loss) değişimini gösteren grafik.

#Matematiksel Çözüm:

Backpropagation algoritmasının detaylı matematiksel çözümü ve adım adım türev hesaplamaları için Matematiksel_Cozum.md dosyasına bakınız.

# Kullanılan Kütüphaneler:

NumPy: Nümerik işlemler ve matris operasyonları için.
Matplotlib: Veri görselleştirme için.
Nasıl Çalıştırılır:

# Bu GitHub deposunu klonlayın:

Bash

git clone https://github.com/csm34/Backpropagation.git
Gerekli kütüphaneleri yükleyin:

Bash

pip install numpy matplotlib
Jupyter Notebook'u başlatın ve Backpropagation.ipynb dosyasını açın.

Kod hücrelerini sırayla çalıştırın.
