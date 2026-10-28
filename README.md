# Drone Tespiti

OpenCV ile videodaki hareketli hedefleri bulan ve **DRONE / KUŞ / UÇAK** olarak ayıran bir görüntü işleme projesi.

## Nasıl çalışır?
1. **Arka plan çıkarma** (MOG2) ile hareketli nesneler bulunur.
2. **Morfolojik işlemler** (erosion, dilation, closing) ile gürültü temizlenir, parçalı kutular birleştirilir.
3. **Mesafe tabanlı takip** ile her hedefin yörüngesi ve kutu geçmişi tutulur.
4. **Kural tabanlı sınıflandırma:**
   - Kuş: kanat çırpma (yükseklik osilasyonu) veya dikey/dar form
   - Uçak: yüksek hızlı, doğrusal rota ve yatay ince gövde
   - Drone: diğer katı gövdeli hedefler

## Kurulum ve çalıştırma
```
pip install -r requirements.txt
python main.py            # test.mp4 üzerinde tespit
python simulate_test.py   # videoya yapay kuş ve uçak ekleyen test modu (b: kuş, a: uçak, q: çıkış)
```
Proje klasörüne kendi videonu `test.mp4` adıyla koyman gerekir (depoda video yoktur).
