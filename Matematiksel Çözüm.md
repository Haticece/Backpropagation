### Backpropagation Algoritması - Matematiksel Çözüm
Bu doküman, Backpropagation.ipynb notebook'unda uygulanan Backpropagation algoritmasının detaylı matematiksel çözümünü sunmaktadır.

## Adım 1: İleri Besleme (Forward Propagation):

# Gizli Katman 1:
z1 = W1 * X + b1 (Matris çarpımı ve toplama)
a1 = sigmoid(z1) (Aktivasyon fonksiyonu uygulanması)

# Gizli Katman 2:
z2 = W2 * a1 + b2
a2 = sigmoid(z2)
# Çıkış Katmanı:
z3 = W3 * a2 + b3
a3 = sigmoid(z3) (Tahmin edilen çıktı)

## Adım 2: Hata Fonksiyonu (Loss Function):

Bu örnekte, Mean Squared Error (MSE) kullanılmaktadır:
L = 0.5 * (Y - a3)^2 (Burada Y gerçek çıktı değeridir)

## Adım 3: Geriye Yayılım (Backpropagation) ile Gradyan Hesaplama:

# Zincir Kuralı: Türevin zincir kuralı kullanılarak hata fonksiyonunun ağırlıklara ve biaslara göre kısmi türevleri hesaplanır.

# Çıkış Katmanı:
d_a3 = -(Y - a3) (Hata fonksiyonunun a3'e göre türevi)
d_z3 = d_a3 * sigmoid'(z3) (Aktivasyon fonksiyonunun türevi: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)))
d_W3 = a2.T * d_z3 (W3'ün türevi)
d_b3 = d_z3 (b3'ün türevi)

# Gizli Katman 2:
d_a2 = d_z3 * W3.T
d_z2 = d_a2 * sigmoid'(z2)
d_W2 = a1.T * d_z2
d_b2 = d_z2

# Gizli Katman 1:
d_a1 = d_z2 * W2.T
d_z1 = d_a1 * sigmoid'(z1)
d_W1 = X.T * d_z1
d_b1 = d_z1

## Adım 4: Ağırlıkların ve Biasların Güncellenmesi:

W3 = W3 - learning_rate * d_W3
b3 = b3 - learning_rate * d_b3
W2 = W2 - learning_rate * d_W2
b2 = b2 - learning_rate * d_b2
W1 = W1 - learning_rate * d_W1   
b1 = b1 - learning_rate * d_b1

Bu dokümanda kullanılan tüm değişkenler ve işlemler Backpropagation.ipynb notebook'undaki kod ile uyumludur. Bu sayede kod ve matematiksel çözüm arasında kolayca ilişki kurulabilir.
