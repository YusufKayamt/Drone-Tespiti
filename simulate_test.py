import cv2
import numpy as np
import os
import math
from collections import deque
from main import classify_target, SimpleTracker, merge_close_boxes

class SimulationScenario:
    """
    Test amacıyla videoya yapay ve gerçekçi Kuş ve Uçak hedefleri enjekte eder.
    Böylece harici bir video aramaya gerek kalmadan sistemin 3 farklı hedefi
    (Drone, Kuş, Uçak) nasıl ayırt ettiğini canlı olarak görebilirsiniz.
    """
    def __init__(self):
        # Kuş parametreleri
        self.bird_x = 50.0
        self.bird_y = 180.0
        self.bird_speed_x = 3.5
        self.bird_speed_y = 0.5
        self.bird_wing_phase = 0.0
        self.bird_active = True
        
        # Uçak parametreleri (Yüksek irtifa, doğrusal, yüksek hız)
        self.plane_x = 980.0
        self.plane_y = 75.0
        self.plane_speed = 9.0
        self.plane_active = True

    def inject_objects(self, frame, frame_idx):
        h, w = frame.shape[:2]
        
        # 1. KANAT ÇIRPAN KUŞ ÇİZİMİ VE HAREKETİ
        if self.bird_active:
            self.bird_x += self.bird_speed_x
            self.bird_y += math.sin(frame_idx * 0.05) * 1.2
            self.bird_wing_phase += 0.45
            
            # Ekrandan çıkarsa başa sar
            if self.bird_x > w + 50:
                self.bird_x = -30
                self.bird_y = 180.0
                
            bx = int(self.bird_x)
            by = int(self.bird_y)
            
            # Kanat çırpma yüksekliği (dikey kanat çırpma osilasyonu)
            wing_span_y = int(math.sin(self.bird_wing_phase) * 32)
            
            # Kuş gövdesi (koyu gri silüet)
            cv2.ellipse(frame, (bx, by), (18, 10), 0, 0, 360, (25, 25, 25), -1)
            # Sol kanat (dikey yukarı-aşağı çırpma)
            wing_pts_left = np.array([
                [bx - 6, by],
                [bx - 12, by + wing_span_y],
                [bx + 4, by]
            ], np.int32)
            cv2.fillPoly(frame, [wing_pts_left], (20, 20, 20))
            # Sağ kanat (dikey yukarı-aşağı çırpma)
            wing_pts_right = np.array([
                [bx - 2, by],
                [bx + 10, by + wing_span_y],
                [bx + 8, by]
            ], np.int32)
            cv2.fillPoly(frame, [wing_pts_right], (20, 20, 20))

        # 2. DOĞRUSAL VE HIZLI UÇAN UÇAK ÇİZİMİ VE HAREKETİ
        if self.plane_active:
            self.plane_x -= self.plane_speed
            # Ekrandan çıkarsa başa sar
            if self.plane_x < -80:
                self.plane_x = w + 60
                
            px = int(self.plane_x)
            py = int(self.plane_y)
            
            # Uçak gövdesi (İnce ve uzun yatay form)
            cv2.ellipse(frame, (px, py), (32, 6), 0, 0, 360, (50, 50, 50), -1)
            # Kanatlar (Geniş delta form)
            plane_wings = np.array([
                [px - 5, py - 18],
                [px + 8, py],
                [px - 5, py + 18],
                [px - 2, py]
            ], np.int32)
            cv2.fillPoly(frame, [plane_wings], (45, 45, 45))
            # Kuyruk
            plane_tail = np.array([
                [px + 22, py - 12],
                [px + 28, py],
                [px + 20, py]
            ], np.int32)
            cv2.fillPoly(frame, [plane_tail], (45, 45, 45))

        return frame

def main():
    video_path = 'test.mp4'
    if not os.path.exists(video_path):
        print(f"HATA: '{video_path}' bulunamadı.")
        return

    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    tracker = SimpleTracker(max_distance=120, max_trace=60)
    sim = SimulationScenario()
    
    print("=" * 65)
    print("      DRONE / KUS / UCAK AYRIMI TEST VE SIMULASYON MODU")
    print("=" * 65)
    print("Kontroller:")
    print("  'b' tuşu: Yapay Kuşu Aç / Kapat")
    print("  'a' tuşu: Yapay Uçağı Aç / Kapat")
    print("  'q' tuşu: Çıkış yap")
    print("=" * 65)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # Döngüsel test için video bittiğinde başa sar
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_idx += 1
        frame = cv2.resize(frame, (1024, 576))
        
        # Test nesnelerini (Kuş ve Uçak) görüntüye ekle
        frame = sim.inject_objects(frame, frame_idx)
        
        # Arka plan çıkarıcıyı uygula
        fg_mask = bg_subtractor.apply(frame)
        
        # Morfolojik temizleme ve birleştirme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 15))
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=2)
        
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 30))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Yapay uzaktaki uçak veya kuşu da yakalamak için eşik 300 piksel
            if area > 300:
                raw_boxes.append(cv2.boundingRect(cnt))
                
        merged_boxes = merge_close_boxes(raw_boxes, max_dist_x=120, max_dist_y=60)
        
        detections = []
        for (x, y, w, h) in merged_boxes:
            cX = x + w // 2
            cY = y + h // 2
            detections.append({'box': (x, y, w, h), 'centroid': (cX, cY)})
            
        tracks = tracker.update(detections)
        
        drone_count = 0
        kus_count = 0
        ucak_count = 0
        
        for track_id, track_data in tracks.items():
            trace = track_data['centroids']
            x, y, w, h = track_data['box']
            label_name = track_data['label']
            color = track_data['color']
            ar = track_data['ar']
            
            if label_name == "DRONE": drone_count += 1
            elif label_name == "KUS": kus_count += 1
            elif label_name == "UCAK": ucak_count += 1
            
            # Kutu ve etiket çiz
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{label_name} - {ar:.2f}"
            text_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Yörünge izi
            for i in range(1, len(trace)):
                if trace[i - 1] is None or trace[i] is None:
                    continue
                thickness = max(1, int(np.sqrt(64 / float(len(trace) - i + 1)) * 2))
                cv2.line(frame, trace[i - 1], trace[i], color, thickness)
                
        # Bilgi Paneli (HUD Overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 12), (540, 56), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"DRONE: {drone_count}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"KUS: {kus_count}", (160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"UCAK: {ucak_count}", (275, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        cv2.putText(frame, f"[B: Kus | A: Ucak]", (395, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Drone / Kus / Ucak Ayrımı Simülasyonu", frame)
        cv2.imshow("Arka Plan Maskesi", fg_mask)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            sim.bird_active = not sim.bird_active
            print(f"Kuş durumu: {'Aktif' if sim.bird_active else 'Devre Dışı'}")
        elif key == ord('a'):
            sim.plane_active = not sim.plane_active
            print(f"Uçak durumu: {'Aktif' if sim.plane_active else 'Devre Dışı'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

