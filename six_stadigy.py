"""
批次處理：讓所有圖片跑過六個策略
每張圖片會產生 6 個不同策略的輸出結果
優化版本 - 包含詳細進度顯示
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import time
from quality_assessment import QualityAssessment

from config import Config
# ============================================
# 六種增強策略
# ============================================

class EnhancementStrategies:
    """六種不同的影像增強策略"""
    
    @staticmethod
    def guided_filter(I, p, r, eps):
        """引導濾波 - 核心演算法"""
        I = I.astype(np.float64)
        p = p.astype(np.float64)
        
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        
        q = mean_a * I + mean_b
        return q
    
    @staticmethod
    def estimate_atmospheric_light(image, min_size=1):
        """
        使用四叉樹方法估計大氣光值
        
        參數:
            image: 輸入圖像 (H x W x 3), 值域 [0, 1]
            min_size: 最小區域大小
        
        返回:
            atmosphere: 大氣光值 (3,) - RGB 向量
        """
        h, w, c = image.shape
        
        # 初始化
        max_Q = -np.inf
        max_RGB = np.array([0.0, 0.0, 0.0])
        
        # 使用堆疊代替遞迴
        stack = [{'block': image, 'nRows': h, 'nCols': w}]
        
        while stack:
            current = stack.pop()
            block = current['block']
            nRows = current['nRows']
            nCols = current['nCols']
            
            # 到達最小尺寸
            if nRows <= min_size or nCols <= min_size:
                Q = EnhancementStrategies.compute_Q(block)
                brightest_RGB = EnhancementStrategies.get_brightest_pixel(block)
                
                if Q > max_Q:
                    max_Q = Q
                    max_RGB = brightest_RGB
            else:
                # 分割為四個區域
                mid_row = nRows // 2
                mid_col = nCols // 2
                
                block1 = block[:mid_row, :mid_col, :]
                block2 = block[:mid_row, mid_col:, :]
                block3 = block[mid_row:, :mid_col, :]
                block4 = block[mid_row:, mid_col:, :]
                
                # 計算四個區域的 Q 值
                Q1 = EnhancementStrategies.compute_Q(block1)
                Q2 = EnhancementStrategies.compute_Q(block2)
                Q3 = EnhancementStrategies.compute_Q(block3)
                Q4 = EnhancementStrategies.compute_Q(block4)
                
                # 找到 Q 值最大的區域
                Q_values = [Q1, Q2, Q3, Q4]
                max_idx = np.argmax(Q_values)
                
                # 只將 Q 值最大的區域放回堆疊
                if max_idx == 0:
                    stack.append({'block': block1, 'nRows': mid_row, 'nCols': mid_col})
                elif max_idx == 1:
                    stack.append({'block': block2, 'nRows': mid_row, 'nCols': nCols - mid_col})
                elif max_idx == 2:
                    stack.append({'block': block3, 'nRows': nRows - mid_row, 'nCols': mid_col})
                else:
                    stack.append({'block': block4, 'nRows': nRows - mid_row, 'nCols': nCols - mid_col})
        
        return max_RGB

    @staticmethod
    def compute_Q(block):
        """
        計算區域的 Q 值評分
        
        參數:
            block: 圖像區域 (H x W x 3)
        
        返回:
            Q: 評分值
        """
        nRows, nCols, _ = block.shape
        n = nRows * nCols
        
        I_r = block[:, :, 0]
        I_g = block[:, :, 1]
        I_b = block[:, :, 2]
        
        # 第一項: 亮度平均
        term1 = (np.sum(I_r) + np.sum(I_g) + np.sum(I_b)) / (3 * n)
        
        # 第二項: 色彩對比項
        term2 = (np.sum(I_b) + np.sum(I_g) - 2 * np.sum(I_r)) / n
        
        # 第三項: 色彩變異項
        mean_r = np.mean(I_r)
        mean_g = np.mean(I_g)
        mean_b = np.mean(I_b)
        var_r = np.sum((I_r - mean_r) ** 2) / n
        var_g = np.sum((I_g - mean_g) ** 2) / n
        var_b = np.sum((I_b - mean_b) ** 2) / n
        term3 = (var_r + var_g + var_b) / 3
        
        # 第四項: 邊緣數量
        gray_img = cv2.cvtColor((block * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_img, 50, 150)
        edge_density = np.sum(edges > 0) / n
        term4 = edge_density
        
        # 合成 Q 值
        Q = term1 + term2 - term3 - term4
        
        return Q
    
    @staticmethod
    def get_brightest_pixel(block):
        """獲取區塊中最亮的像素 RGB 值"""
        nRows, nCols, _ = block.shape
        brightness = np.sum(block, axis=2)
        max_idx = np.unravel_index(np.argmax(brightness), brightness.shape)
        return block[max_idx[0], max_idx[1], :]
    
    @staticmethod
    def estimate_transmission(img, atmospheric_light, omega, guided_radius, eps):
        """估計透射圖"""
        normalized = img / (atmospheric_light.reshape(1, 1, 3) + 1e-6)
        dark_channel = np.min(normalized, axis=2)
        
        transmission = 1 - omega * dark_channel
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # 引導濾波細化
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        transmission_refined = EnhancementStrategies.guided_filter(gray, transmission, guided_radius, eps)
        
        return np.clip(transmission_refined, 0.1, 1.0)
    
    @staticmethod
    def restore_image(img, atmospheric_light, transmission):
        """影像復原"""
        result = np.zeros_like(img)
        for i in range(3):
            result[:,:,i] = (img[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        return np.clip(result, 0, 1)
    
    @staticmethod
    def enhance_contrast(img, L_low, L_high):
        """對比度增強"""
        result = np.zeros_like(img)
        for i in range(3):
            channel = img[:,:,i]
            p_low = np.percentile(channel, L_low)
            p_high = np.percentile(channel, L_high)
            result[:,:,i] = np.clip((channel - p_low) / (p_high - p_low + 1e-6), 0, 1)
        return result
    
    @staticmethod
    def apply_clahe(img, clip_limit=2.0):
        """CLAHE 對比度增強"""
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return result
    
    @staticmethod
    def white_balance(img, percentile=5):
        """白平衡"""
        result = np.zeros_like(img)
        for i in range(3):
            channel = img[:,:,i]
            p_low = np.percentile(channel, percentile)
            p_high = np.percentile(channel, 100 - percentile)
            result[:,:,i] = np.clip((channel - p_low) / (p_high - p_low + 1e-6), 0, 1)
        return result
    
    @staticmethod
    def gamma_correction(img, gamma=1.2):
        """Gamma 校正"""
        return np.power(img, gamma)
    
    # ========================================
    # 六種完整策略
    # ========================================
    
    @classmethod
    def strategy1_strong_dehazing(cls, img):
        """策略1：強力去霧 - 適合嚴重霧霾影像"""
        atmospheric_light = cls.estimate_atmospheric_light(img, min_size=1)
        transmission = cls.estimate_transmission(img, atmospheric_light, omega=0.3, guided_radius=20, eps=5e-1)
        restored = cls.restore_image(img, atmospheric_light, transmission)
        enhanced = cls.enhance_contrast(restored, L_low=5, L_high=98)
        enhanced = cls.apply_clahe(enhanced, clip_limit=3.0)
        enhanced = cls.gamma_correction(enhanced, gamma=1.5)
        return enhanced
    
    @classmethod
    def strategy2_medium_dehazing(cls, img):
        """策略2：中度去霧 - 平衡版本"""
        atmospheric_light = cls.estimate_atmospheric_light(img, min_size=1)
        transmission = cls.estimate_transmission(img, atmospheric_light, omega=0.5, guided_radius=15, eps=5e-1)
        restored = cls.restore_image(img, atmospheric_light, transmission)
        enhanced = cls.enhance_contrast(restored, L_low=15, L_high=95)
        enhanced = cls.apply_clahe(enhanced, clip_limit=2.0)
        return enhanced
    
    @classmethod
    def strategy3_light_dehazing(cls, img):
        """策略3：輕度去霧 - 保留自然感"""
        atmospheric_light = cls.estimate_atmospheric_light(img, min_size=1)
        transmission = cls.estimate_transmission(img, atmospheric_light, omega=0.7, guided_radius=10, eps=1e-1)
        restored = cls.restore_image(img, atmospheric_light, transmission)
        enhanced = cls.enhance_contrast(restored, L_low=20, L_high=85)
        enhanced = cls.white_balance(enhanced, percentile=2)
        return enhanced
    
    @classmethod
    def strategy4_clahe_enhancement(cls, img):
        """策略4：CLAHE 增強 - 適合低對比度影像"""
        enhanced = cls.apply_clahe(img, clip_limit=4.0)
        enhanced = cls.enhance_contrast(enhanced, L_low=10, L_high=95)
        enhanced = cls.white_balance(enhanced, percentile=3)
        enhanced = cls.gamma_correction(enhanced, gamma=1.3)
        return enhanced
    
    @classmethod
    def strategy5_white_balance(cls, img):
        """策略5：白平衡主導 - 適合色偏影像"""
        enhanced = cls.white_balance(img, percentile=2)
        enhanced = cls.enhance_contrast(enhanced, L_low=15, L_high=90)
        enhanced = cls.apply_clahe(enhanced, clip_limit=1.5)
        enhanced = cls.gamma_correction(enhanced, gamma=1.2)
        return enhanced
    
    @classmethod
    def strategy6_histogram_eq(cls, img):
        """策略6：直方圖均衡 - 適合暗影像"""
        enhanced = cls.enhance_contrast(img, L_low=5, L_high=98)
        enhanced = cls.apply_clahe(enhanced, clip_limit=3.5)
        enhanced = cls.gamma_correction(enhanced, gamma=1.4)
        return enhanced


# ============================================
# 色偏偵測與校正
# ============================================

def detect_image_type(img):
    """偵測影像類型（綠偏/藍偏/正常）"""
    mean_rgb = img.mean(axis=(0, 1))
    r, g, b = mean_rgb
    
    if g > r and g > b and (g - r) > 0.05:
        return "greenish"
    elif b > r and b > g and (b - r) > 0.05:
        return "bluish"
    else:
        return "normal"


def color_correction(img, image_type):
    """根據影像類型進行色偏校正"""
    if image_type == "greenish":
        # 綠偏校正：降低綠色通道
        corrected = img.copy()
        corrected[:,:,1] = corrected[:,:,1] * 0.85
        corrected = np.clip(corrected, 0, 1)
        return corrected
    
    elif image_type == "bluish":
        # 藍偏校正：降低藍色通道
        corrected = img.copy()
        corrected[:,:,2] = corrected[:,:,2] * 0.85
        corrected = np.clip(corrected, 0, 1)
        return corrected
    
    else:
        
        return img


# ============================================
# 主處理函數（優化版）
# ============================================

def process_all_images_all_strategies(input_folder, output_base_folder):
    """
    處理所有影像，每張影像跑過六個策略
    優化版本 - 包含詳細進度顯示
    
    參數:
        input_folder: 輸入影像資料夾
        output_base_folder: 輸出基礎資料夾
    """
    
    # 創建輸出資料夾
    input_path = Path(input_folder)
    output_path = Path(output_base_folder)
    output_path.mkdir(parents=True, exist_ok=True)  
    strategies = [
        ('strong_dehazing', EnhancementStrategies.strategy1_strong_dehazing, '強力去霧'),
        ('medium_dehazing', EnhancementStrategies.strategy2_medium_dehazing, '中度去霧'),
        ('light_dehazing', EnhancementStrategies.strategy3_light_dehazing, '輕度去霧'),
        ('clahe_enhancement', EnhancementStrategies.strategy4_clahe_enhancement, 'CLAHE增強'),
        ('white_balance', EnhancementStrategies.strategy5_white_balance, '白平衡主導'),
        ('histogram_eq', EnhancementStrategies.strategy6_histogram_eq, '直方圖均衡'),
    ]
    
    # 創建輸出資料夾（所有結果放在同一個資料夾）
    print("\n📁 創建輸出資料夾...")
    print(f"   ✓ {output_path}")
    print(f"   所有策略結果將放在同一個資料夾，檔名格式: 原檔名_s1.png, 原檔名_s2.png, ...")
    
    # 獲取所有影像檔案
    print("\n🔍 搜尋影像檔案...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(ext)))
        image_files.extend(list(input_path.glob(ext.upper())))
    
    if len(image_files) == 0:
        print(f"❌ 在 {input_folder} 中找不到影像檔案！")
        return
    

    
    # 記錄處理結果
    log_data = []
    
    # 統計計數器
    stats = {
        'total_images': len(image_files),
        'processed_images': 0,
        'failed_images': 0,
        'total_outputs': 0,
        'successful_outputs': 0,
        'failed_outputs': 0,
        'image_types': {'greenish': 0, 'bluish': 0, 'normal': 0}
    }
    
    # 開始處理
    start_time = time.time()
    
    # 外層進度條：影像
    with tqdm(total=len(image_files), desc="🖼️  處理影像", unit="張", ncols=100) as pbar_images:
        
        for img_idx, img_path in enumerate(image_files):
            img_start_time = time.time()
            img_success = False
            
            try:
                # 讀取影像
                img = cv2.imread(str(img_path))
                if img is None:
                    pbar_images.write(f"  無法讀取: {img_path.name}")
                    stats['failed_images'] += 1
                    pbar_images.update(1)
                    continue
                
                # 轉換為 RGB 並正規化
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # 偵測影像類型
                image_type = detect_image_type(img)
                stats['image_types'][image_type] += 1
                
                # 色偏校正
                img_corrected = color_correction(img, image_type)
                
                # 內層進度條：策略（只在第一張圖時顯示）
                strategy_desc = f"   └─ 應用策略 ({img_path.name[:20]}...)"
                
                with tqdm(total=len(strategies), desc=strategy_desc, 
                         unit="策略", leave=False, ncols=100) as pbar_strategies:
                    
                    img_strategy_success = 0
                    
                    # 對每個策略進行處理
                    for strategy_name, strategy_func, strategy_desc_text in strategies:
                        try:
                            # 應用策略
                            enhanced = strategy_func(img_corrected)
                            
                            # 轉換回 uint8
                            enhanced_uint8 = (enhanced * 255).astype(np.uint8)
                            enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
                        #     score, _ = QualityAssessment.comprehensive_assessment(
                        #     enhanced_bgr,
                        #         weights={
                        #             'contrast': 0.20,
                        #             'sharpness': 0.20,
                        #             'entropy': 0.15,
                        #             'saturation': 0.15,
                        #             'brightness': 0.10,
                        #             'edge_density': 0.10,
                        #             'colorfulness': 0.05,
                        #             'naturalness': 0.05
                        #         }
                        # )
                            # 儲存結果 - 使用檔名後綴（例如：image_s1.png, image_s2.png）
                            output_file = output_path / f"{img_path.stem}_{strategy_name}.png"
                            cv2.imwrite(str(output_file), enhanced_bgr)
                            
                            # 記錄成功
                            log_data.append({
                                'filename': img_path.name,
                                'image_type': image_type,
                                'strategy': strategy_name,
                                'strategy_desc': strategy_desc_text,
                                'status': 'success',
                                'output_path': str(output_file),
                                'processing_time': f"{time.time() - img_start_time:.2f}s",
                                # 'score': f"{score:.2f}"
                            })
                            
                            stats['successful_outputs'] += 1
                            img_strategy_success += 1
                            
                        except Exception as e:
                            # 記錄失敗
                            error_msg = str(e)[:50]
                            pbar_images.write(f"   ✗ {strategy_desc_text} 失敗: {error_msg}")
                            
                            log_data.append({
                                'filename': img_path.name,
                                'image_type': image_type,
                                'strategy': strategy_name,
                                'strategy_desc': strategy_desc_text,
                                'status': 'failed',
                                'output_path': f'Error: {error_msg}',
                                'processing_time': 'N/A'
                                
                            })
                            
                            stats['failed_outputs'] += 1
                        
                        finally:
                            pbar_strategies.update(1)
                
                # 更新統計
                if img_strategy_success > 0:
                    stats['processed_images'] += 1
                    img_success = True
                else:
                    stats['failed_images'] += 1
                
                stats['total_outputs'] += len(strategies)
                
                # 計算處理時間
                img_time = time.time() - img_start_time
                
                # 更新外層進度條的描述
                elapsed = time.time() - start_time
                avg_time = elapsed / (img_idx + 1)
                remaining = avg_time * (len(image_files) - img_idx - 1)
                
                pbar_images.set_postfix({
                    '成功': f"{stats['processed_images']}/{stats['total_images']}",
                    '本張': f"{img_time:.1f}s",
                    '剩餘': f"{remaining/60:.1f}m"
                })
                
            except Exception as e:
                pbar_images.write(f"✗ 處理失敗: {img_path.name} - {str(e)}")
                stats['failed_images'] += 1
            
            finally:
                pbar_images.update(1)
    
    # 儲存處理記錄
    print("\n💾 儲存處理記錄...")
    df = pd.DataFrame(log_data)
    csv_path = output_path / 'processing_log.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ 記錄已儲存: {csv_path}")
    
   

    
    
    print(f"\n📁 輸出位置:")
    print(f"   {output_path}")
    
    
   


# ============================================
# 使用範例
# ============================================

if __name__ == "__main__":
    # 設定路徑
    INPUT_FOLDER = r"D:\rop\Jamaica\Jamaica"
    OUTPUT_FOLDER = r"D:\rop\Jamaica\output_six_strategies"

    print(f"\n📂 輸入資料夾: {INPUT_FOLDER}")
    print(f"📂 輸出資料夾: {OUTPUT_FOLDER}\n")
    
    # 檢查輸入資料夾
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 輸入資料夾不存在: {INPUT_FOLDER}")
        print("請修改 INPUT_FOLDER 變數為正確的路徑")
    else:
        # 執行處理
        process_all_images_all_strategies(INPUT_FOLDER, OUTPUT_FOLDER)