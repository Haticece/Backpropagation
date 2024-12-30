# Yapay Sinir Ağlarında Arka Yayılım (Backpropagation)

## Genel Bakış
Bu repo, **Backpropagation (Arka Yayılım)** algoritmasının yapay sinir ağlarını eğitmek için adım adım nasıl uygulanacağını gösteren bir Jupyter Notebook içermektedir. Bu notebook, algoritmanın temel işleyişini, bir sinir ağı modelinin başlatılmasından başlayarak, gradyanların hesaplanmasına ve kayıp fonksiyonunu minimize etmek için ağırlıkların nasıl güncellendiğine kadar olan süreci detaylı bir şekilde göstermektedir. Bu yöntem, derin öğrenmede sinir ağlarını optimize etmek için temel bir tekniktir.

## Temel Kavramlar
- **Yapay Sinir Ağları**: Veriyi işleyen ve denetimli öğrenme yoluyla tahminlerde bulunan düğüm (nöron) ağlarıdır.
- **Backpropagation (Arka Yayılım)**: Yapay sinir ağlarını eğitmek için kullanılan denetimli öğrenme algoritmasıdır. Ağırlıklar, zincir kuralı ile hesaplanan gradyanlar yardımıyla güncellenir.
- **Gradyan İnişi (Gradient Descent)**: Kayıp fonksiyonunu minimize etmek için kullanılan bir optimizasyon algoritmasıdır.
- **Kayıp Fonksiyonu (Loss Function)**: Tahmin edilen değer ile gerçek etiket arasındaki farkı ölçen bir fonksiyondur. Amaç bu hatayı minimize etmektir.

## Arka Yayılımın Çalışma Prensibi
Arka yayılım, kayıp fonksiyonunun gradyanlarını (veya türevlerini) sinir ağının ağırlıkları ve biaslarıyla hesaplar. Bu gradyanlar ağ boyunca geriye doğru yayılır ve ağırlıklar güncellenir.

### Adımlar:
1. **İleri Yayılım (Forward Pass)**: Girdi verisi ağ boyunca geçirilerek çıktı elde edilir.
2. **Kayıp Hesaplama**: Tahmin edilen çıktı ile gerçek etiket arasındaki fark, bir kayıp fonksiyonu kullanılarak hesaplanır (örneğin, Ortalama Kare Hata veya Çapraz Entropi).
3. **Geriye Yayılım (Backward Pass)**: Kayıp fonksiyonunun türevleri, ağın ağırlıkları ile hesaplanır ve zincir kuralı ile geriye doğru yayılır.
4. **Ağırlık Güncellemesi**: Ağırlıklar, gradyan inişi gibi bir optimizasyon algoritmasıyla güncellenir ve kayıp fonksiyonu minimize edilir.

## Gereksinimler
Notebook'u çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız olacak:
- `numpy` (sayısal hesaplamalar için)
- `matplotlib` (grafikler ve görselleştirme için)
- `pandas` (isteğe bağlı, veri işleme için)

Gerekli kütüphaneleri `pip` ile yükleyebilirsiniz:

```bash
pip install numpy matplotlib pandas
```

## Kullanım
1. Reposunu klonlayın ya da notebook dosyasını bilgisayarınıza indirin.
2. Python ortamında Jupyter notebook'u açın.
3. Her hücreyi sırayla çalıştırarak arka yayılım algoritmasını çalıştırın ve sinir ağının ağırlıklarını nasıl güncellediğini görün.
4. Öğrenme sürecindeki kayıp grafiğini analiz ederek ağın nasıl öğrendiğini gözlemleyin.

## Örnek Kod
İşte notebook'ta yer alan basit bir backpropagation uygulamasına dair örnek bir kod:

```python
import numpy as np

# Örnek ileri yayılım (1 katmanlı sinir ağı)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Örnek arka yayılım (gradyan hesaplama)
def backpropagation(X, y, weights, learning_rate):
    # İleri yayılım
    output = sigmoid(np.dot(X, weights))
    
    # Hata (kayıp) hesaplama
    error = y - output
    
    # Geriye yayılım (gradyan hesaplama)
    gradients = np.dot(X.T, error * output * (1 - output))
    
    # Ağırlık güncellemesi
    weights += learning_rate * gradients
    return weights
```

Bu basit örnekte algoritma, çıktıyı hesaplar, hatayı belirler, hatayı geri yayarak gradyanları hesaplar ve ardından ağırlıkları günceller.

## Sonuçlar ve Değerlendirme
Notebook'ta, eğitim sürecindeki kayıp fonksiyonu görselleştirilmektedir. Model, backpropagation kullanarak eğitim aldıkça, kayıp değeri zamanla azalır ve bu da ağın doğru tahminler yapmaya başladığını gösterir.

## Sonuç
Bu notebook, **Backpropagation** algoritmasının nasıl çalıştığını gösteren pratik bir örnek sunmaktadır. Derin öğrenme modelleri ile çalışan herkes için bu algoritmanın anlaşılması önemlidir. Arka yayılım uygulayarak ağın ağırlıklarını optimize edebilir ve daha doğru tahminler yapılmasını sağlayabilirsiniz.
