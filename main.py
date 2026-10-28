import cv2
import numpy as np
import os
import math
from collections import deque

def classify_target(w, h, history_boxes=None, history_centroids=None):
    """
    Hedefin morfolojik ve kinematik özelliklerine göre sınıflandırma yapar:
    - KUS (Mavi): Kanat çırpma osilasyonu (yükseklik dalgalanması) veya dar/dikey form (aspect_ratio < 1.25)
    - UCAK (Sarı): Yüksek hızda doğrusal rota (linearity > 0.94, speed > 7 px/frame) ve yatay ince gövde (aspect_ratio >= 1.75)
    - DRONE (Kırmızı): Katı gövde formu, yatay genişlik (aspect_ratio >= 1.25), dönüş ve asılı kalma kabiliyeti
    """
    inst_ar = w / float(h)
    
    # Zaman bazlı yumuşatılmış en-boy oranı (anlık dönüşlerde titremeyi engeller)
    if history_boxes and len(history_boxes) > 0:
        recent_ars = [b[0] / float(b[1]) for b in list(history_boxes)[-15:]]
        smoothed_ar = float(np.mean(recent_ars))
    else:
        smoothed_ar = inst_ar

    # 1. Gerçek Kanat Çırpma (Flapping / Yön Değişimli Osilasyon) Kontrolü
    is_flapping = False
    if history_boxes and len(history_boxes) >= 8:
        recent_h = [box[1] for box in list(history_boxes)[-10:]]
        diffs = [recent_h[i] - recent_h[i-1] for i in range(1, len(recent_h))]
        sig_diffs = [d for d in diffs if abs(d) >= 2]
        if len(sig_diffs) >= 4:
            # Yükseklik farklarının işaret değişim sayısı (yukarı-aşağı kanat vuruşu)
            sign_changes = sum(1 for i in range(1, len(sig_diffs)) if (sig_diffs[i] * sig_diffs[i-1]) < 0)
            if sign_changes >= 3 and smoothed_ar < 1.40:
                is_flapping = True

    # 2. Uçak (Doğrusal yüksek hızlı rota) Kontrolü
    is_airplane = False
    if history_centroids and len(history_centroids) >= 12:
        pts = np.array(list(history_centroids)[-20:])
        direct_dist = np.linalg.norm(pts[-1] - pts[0])
        step_diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path_dist = np.sum(step_diffs)
        linearity = direct_dist / (path_dist + 1e-5)
        avg_speed = np.mean(step_diffs[-6:]) if len(step_diffs) >= 6 else 0
        
        # Uçak: Doğrusal uçuş, yüksek hız ve yatay ince-uzun gövde
        if linearity > 0.94 and avg_speed > 7.0 and smoothed_ar >= 1.75:
            is_airplane = True

    if is_flapping or smoothed_ar < 1.25:
        return "KUS", (255, 0, 0), smoothed_ar # Mavi
    elif is_airplane:
        return "UCAK", (0, 215, 255), smoothed_ar # Sarı / Turuncu
    else:
        return "DRONE", (0, 0, 255), smoothed_ar # Kırmızı

class SimpleTracker:
    """
    Mesafe tabanlı çoklu nesne takip ve sınıflandırma algoritması.
    Ağırlık merkezlerini, kutu boyutlarını ve yörünge özelliklerini takip eder.
    """
    def __init__(self, max_distance=120, max_trace=60):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_trace = max_trace

    def update(self, detections):
        """
        detections: list of dicts [{'box': (x, y, w, h), 'centroid': (cX, cY)}, ...]
        """
        updated_tracks = {}
        
        for det in detections:
            centroid = det['centroid']
            box = det['box']
            min_dist = float('inf')
            best_id = -1
            
            # En yakın geçmiş noktayı bul
            for track_id, track_data in self.tracks.items():
                trace = track_data['centroids']
                if not trace: continue
                last_pt = trace[-1]
                dist = math.hypot(centroid[0] - last_pt[0], centroid[1] - last_pt[1])
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = track_id
            
            if best_id != -1:
                track = self.tracks[best_id]
                track['centroids'].append(centroid)
                track['boxes'].append((box[2], box[3])) # w, h
                
                # Kararlı sınıflandırma (geçmiş ortalaması ile)
                label, color, ar = classify_target(box[2], box[3], track['boxes'], track['centroids'])
                track['label'] = label
                track['color'] = color
                track['ar'] = ar
                track['box'] = box
                
                updated_tracks[best_id] = track
                del self.tracks[best_id]
            else:
                new_centroids = deque(maxlen=self.max_trace)
                new_centroids.append(centroid)
                new_boxes = deque(maxlen=self.max_trace)
                new_boxes.append((box[2], box[3]))
                
                label, color, ar = classify_target(box[2], box[3], new_boxes, new_centroids)
                
                updated_tracks[self.next_id] = {
                    'centroids': new_centroids,
                    'boxes': new_boxes,
                    'label': label,
                    'color': color,
                    'ar': ar,
                    'box': box
                }
                self.next_id += 1
                
        self.tracks = updated_tracks
        return self.tracks

def merge_close_boxes(boxes, max_dist_x=120, max_dist_y=60):
    """
    Birbirine yakın duran ayrık parçaları (örneğin drone gövdesi ve kanat/pervane uçları)
    tek bir çevreleyen kutu (bounding box) altında birleştirir.
    """
    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                x2, y2, w2, h2 = boxes[j]
                
                # X ve Y eksenlerindeki mesafe/boşluk hesabı
                gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                
                if gap_x <= max_dist_x and gap_y <= max_dist_y:
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    x1, y1, w1, h1 = nx, ny, nw, nh
                    skip.add(j)
                    merged = True
            new_boxes.append((x1, y1, w1, h1))
        boxes = new_boxes
    return boxes

def main():
    video_path = 'test.mp4'
    
    # 3. Video dosyası yoksa çökmeyi engelle ve terminale hata yazdır
    if not os.path.exists(video_path):
        print(f"HATA: '{video_path}' bulunamadı.")
        return

    cap = cv2.VideoCapture(video_path)
    
    # 5. BackgroundSubtractorMOG2 ile arka planı çıkarma
    # history=500: arka planın ne kadar sürede güncelleneceği
    # varThreshold=50: pikselin arka plan/ön plan ayrımı için eşik değeri
    # detectShadows=True: gölgeleri gri tonlama olarak (127) işaretler
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    # Hedeflerin yörüngelerini izlemek için tracker nesnesi oluştur
    tracker = SimpleTracker(max_distance=100, max_trace=60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya okunamadı.")
            break
            
        # 4. Görüntüyü işleme sokmadan önce ekrana sığması için 1024x576 boyutlarına getir
        frame = cv2.resize(frame, (1024, 576))
        
        # Arka plan çıkarıcıyı uygula (Siyah-Beyaz maske üretir)
        fg_mask = bg_subtractor.apply(frame)
        
        # 6. Morfolojik işlemler (Erosion ve Dilation) ile kamera ve rüzgar gürültülerini temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Erosion (Aşındırma): Küçük beyaz noktaları (gürültüleri) yok eder
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        # Dilation (Genişletme): Aşınmış asıl nesneleri eski boyutuna getirir ve içlerindeki boşlukları doldurur
        # Drone yatayda geniş olduğu ve kanat eğimine sahip olduğu için uygun bir dikdörtgen kernel kullanıyoruz
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 15))
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=2)
        
        # Gölgeleri (127 gri değeri) maskeden temizlemek için Threshold işlemi (sadece 255 olanları alır)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        
        # Maskeyi bütünleştirmek ve ayrı parçaları birleştirmek için Kapanma (Closing) işlemi
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 30))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # 7. findContours ile temizlenmiş beyaz lekelerin sınırlarını bul
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Çok küçük alanları (çöp, yaprak) yoksay
            if area > 800:
                x, y, w, h = cv2.boundingRect(cnt)
                raw_boxes.append((x, y, w, h))
                
        # Kanat veya pervane parçalarının ayrı algılanmasını önlemek için yakın kutuları birleştir
        merged_boxes = merge_close_boxes(raw_boxes, max_dist_x=120, max_dist_y=60)
        
        detections = []
        for (x, y, w, h) in merged_boxes:
            cX = x + w // 2
            cY = y + h // 2
            detections.append({'box': (x, y, w, h), 'centroid': (cX, cY)})
                    
        # Tracker'ı mevcut karedeki tespitlerle güncelle
        tracks = tracker.update(detections)
        
        drone_count = 0
        kus_count = 0
        ucak_count = 0
        
        # Her takip edilen nesneyi kendi sınıfının renginde ve etiketinde çiz
        for track_id, track_data in tracks.items():
            trace = track_data['centroids']
            x, y, w, h = track_data['box']
            label_name = track_data['label']
            color = track_data['color']
            ar = track_data['ar']
            
            if label_name == "DRONE": drone_count += 1
            elif label_name == "KUS": kus_count += 1
            elif label_name == "UCAK": ucak_count += 1
            
            # Nesnenin etrafına sınıf renginde kutu çiz
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Yazının ekrandan taşmasını engellemek için y eksenini kontrol et
            label = f"{label_name} - {ar:.2f}"
            text_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Hedefin arkasından uzayan yörünge çizgisini sınıf renginde çiz
            for i in range(1, len(trace)):
                if trace[i - 1] is None or trace[i] is None:
                    continue
                thickness = max(1, int(np.sqrt(64 / float(len(trace) - i + 1)) * 2))
                cv2.line(frame, trace[i - 1], trace[i], color, thickness)
                
        # Bilgi Paneli (HUD / Status Bar)
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 12), (370, 52), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"DRONE: {drone_count}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(frame, f"KUS: {kus_count}", (145, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
        cv2.putText(frame, f"UCAK: {ucak_count}", (250, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

        # 9. Orijinal video ve Arka plan maskesi olmak üzere iki farklı pencere göster
        cv2.imshow("Orijinal Video - Yörünge Takibi", frame)
        cv2.imshow("Arka Plan Maskesi", fg_mask)
        
        # 'q' tuşuna basıldığında güvenli çıkış yap
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Kullanıcı tarafından çıkış yapıldı.")
            break

    # Temizlik işlemleri
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
cv2.destroyAllWindows()