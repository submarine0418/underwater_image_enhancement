"""
特徵提取模組
提取影像的多維度特徵用於分類
"""

import cv2
import numpy as np
from skimage import color, feature, filters
from skimage.measure import shannon_entropy
from scipy import stats


class FeatureExtractor:
    """影像特徵提取器"""
    
    @staticmethod
    def extract_color_features(img):
        """
        提取顏色特徵
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            features: 特徵向量
        """
        features = []
        
        # === LAB 色彩空間特徵 ===
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        for c in range(3):
            channel = lab[:, :, c].flatten()
            features.extend([
                np.mean(channel),
                np.std(channel),
                stats.skew(channel),
                stats.kurtosis(channel)
            ])
        
        # === HSV 色彩空間特徵 ===
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        for c in range(3):
            channel = hsv[:, :, c].flatten()
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # === 色偏指標 (Color Cast Factor) ===
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        mean_a = np.mean(a_channel)
        mean_b = np.mean(b_channel)
        M = np.sqrt(mean_a**2 + mean_b**2)
        
        Da = np.mean(np.abs(a_channel - mean_a))
        Db = np.mean(np.abs(b_channel - mean_b))
        D = np.sqrt(Da**2 + Db**2)
        
        CCF = M / (D + 1e-10)
        features.extend([CCF, M, D, mean_a, mean_b])
        
        # === RGB 通道統計 ===
        for c in range(3):
            channel = img[:, :, c].flatten()
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        return np.array(features)
    
    @staticmethod
    def extract_texture_features(img):
        """
        提取紋理特徵
        
        Args:
            img: RGB影像 (0-1 範圍)
        """
        features = []
        
        # 轉灰度
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # === LBP (Local Binary Patterns) ===
        radius = 1
        n_points = 8 * radius
        # 轉換為 uint8 避免警告
        gray_uint8 = (gray * 255).astype(np.uint8)
        lbp = feature.local_binary_pattern(gray_uint8, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
        features.extend(lbp_hist)
        
        # === 灰度共生矩陣 (GLCM) ===
        # 降低解析度以加快計算
        gray_uint8 = (gray * 255).astype(np.uint8)
        gray_reduced = cv2.resize(gray_uint8, (128, 128))
        
        # 計算 GLCM
        glcm = feature.graycomatrix(gray_reduced, 
                                    distances=[1], 
                                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    levels=256,
                                    symmetric=True, 
                                    normed=True)
        
        # 提取 GLCM 特徵
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            values = feature.graycoprops(glcm, prop).flatten()
            features.extend([np.mean(values), np.std(values)])
        
        return np.array(features)
    
    @staticmethod
    def extract_frequency_features(img):
        """
        提取頻率域特徵
        
        Args:
            img: RGB影像 (0-1 範圍)
        """
        features = []
        
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # === DCT 特徵 ===
        dct = cv2.dct(gray)
        
        # DCT 能量分布
        total_energy = np.sum(dct**2)
        
        # 低頻能量（左上角 1/4）
        h, w = dct.shape
        low_freq_energy = np.sum(dct[:h//4, :w//4]**2) / total_energy
        
        # 中頻能量
        mid_freq_energy = np.sum(dct[h//4:h//2, w//4:w//2]**2) / total_energy
        
        # 高頻能量
        high_freq_energy = np.sum(dct[h//2:, w//2:]**2) / total_energy
        
        features.extend([
            low_freq_energy,
            mid_freq_energy,
            high_freq_energy,
            np.mean(np.abs(dct)),
            np.std(np.abs(dct))
        ])
        
        return np.array(features)
    
    @staticmethod
    def extract_edge_features(img):
        """
        提取邊緣特徵
        
        Args:
            img: RGB影像 (0-1 範圍)
        """
        features = []
        
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # === Sobel 邊緣 ===
        # 確保輸入是 float32
        gray_f32 = gray.astype(np.float32)
        sobelx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features.extend([
            np.mean(sobel_magnitude),
            np.std(sobel_magnitude),
            np.max(sobel_magnitude)
        ])
        
        # === Canny 邊緣密度 ===
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # === Laplacian (清晰度) ===
        # 修復：確保使用正確的數據類型
        gray_uint8 = (gray * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F, ksize=3)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            np.var(laplacian)  # 清晰度指標
        ])
        
        return np.array(features)
    
    @staticmethod
    def extract_quality_features(img):
        """
        提取品質相關特徵
        
        Args:
            img: RGB影像 (0-1 範圍)
        """
        features = []
        
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # === 對比度 ===
        contrast = np.std(gray)
        features.append(contrast)
        
        # === 熵 (資訊量) ===
        entropy = shannon_entropy(gray)
        features.append(entropy)
        
        # === 亮度統計 ===
        features.extend([
            np.mean(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75)
        ])
        
        # === 動態範圍 ===
        dynamic_range = np.max(gray) - np.min(gray)
        features.append(dynamic_range)
        
        # === 色彩豐富度 ===
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        saturation = hsv[:, :, 1]
        features.extend([
            np.mean(saturation),
            np.std(saturation)
        ])
        
        # === RMS 對比度 ===
        rms_contrast = np.sqrt(np.mean((gray - np.mean(gray))**2))
        features.append(rms_contrast)
        
        return np.array(features)
    
    @staticmethod
    def extract_all_features(img):
        """
        提取所有特徵
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            features: 完整特徵向量
        """
        features = []
        
        # 提取各類特徵
        try:
            color_feat = FeatureExtractor.extract_color_features(img)
            features.append(color_feat)
        except Exception as e:
            print(f"顏色特徵提取失敗: {e}")
        
        try:
            texture_feat = FeatureExtractor.extract_texture_features(img)
            features.append(texture_feat)
        except Exception as e:
            print(f"紋理特徵提取失敗: {e}")
        
        try:
            freq_feat = FeatureExtractor.extract_frequency_features(img)
            features.append(freq_feat)
        except Exception as e:
            print(f"頻率特徵提取失敗: {e}")
        
        try:
            edge_feat = FeatureExtractor.extract_edge_features(img)
            features.append(edge_feat)
        except Exception as e:
            print(f"邊緣特徵提取失敗: {e}")
        
        try:
            quality_feat = FeatureExtractor.extract_quality_features(img)
            features.append(quality_feat)
        except Exception as e:
            print(f"品質特徵提取失敗: {e}")
        
        # 合併所有特徵
        if len(features) > 0:
            all_features = np.concatenate(features)
            return all_features
        else:
            raise ValueError("無法提取任何特徵")


def test_feature_extraction():
    """測試特徵提取"""
    print("測試特徵提取...")
    
    # 創建測試影像
    test_img = np.random.rand(256, 256, 3)
    
    # 提取特徵
    features = FeatureExtractor.extract_all_features(test_img)
    
    print(f"特徵維度: {len(features)}")
    print(f"特徵範圍: [{np.min(features):.4f}, {np.max(features):.4f}]")
    print(f"特徵均值: {np.mean(features):.4f}")
    print(f"特徵標準差: {np.std(features):.4f}")
    
    # 檢查是否有 NaN 或 Inf
    if np.any(np.isnan(features)):
        print("警告: 特徵中包含 NaN")
    if np.any(np.isinf(features)):
        print("警告: 特徵中包含 Inf")
    
    print("✓ 特徵提取測試完成")


if __name__ == '__main__':
    test_feature_extraction()