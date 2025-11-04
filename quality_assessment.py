"""
無參考影像品質評估 (NR-IQA)
"""

import cv2
import numpy as np
from skimage.measure import shannon_entropy
from scipy import ndimage


class QualityAssessment:
    """無參考影像品質評估器"""
    
    @staticmethod
    def assess_contrast(img):
        """
        評估對比度
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 對比度分數 (0-100)
        """
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # RMS 對比度
        contrast = np.std(gray)
        
        # 正規化到 0-100 (假設好的對比度在 0.1-0.5 範圍)
        score = np.clip(contrast / 0.5 * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_sharpness(img):
        """
        評估清晰度
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 清晰度分數 (0-100)
        """
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Laplacian 方差法
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 正規化到 0-100 (假設好的清晰度在 0-0.5 範圍)
        score = np.clip(sharpness / 0.5 * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_entropy(img):
        """
        評估資訊熵
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 熵分數 (0-100)
        """
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        entropy = shannon_entropy(gray)
        
        # 正規化到 0-100 (假設好的熵在 6-8 範圍)
        score = np.clip((entropy - 4) / 4 * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_saturation(img):
        """
        評估色彩飽和度
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 飽和度分數 (0-100)
        """
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        saturation = hsv[:, :, 1]
        
        # 平均飽和度
        mean_sat = np.mean(saturation)
        
        # 正規化到 0-100 (假設好的飽和度在 0.2-0.8 範圍)
        score = np.clip(mean_sat * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_brightness(img):
        """
        評估亮度適中度
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 亮度分數 (0-100)
        """
        # 轉換到 LAB 空間
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        
        # 計算平均亮度
        mean_brightness = np.mean(L)
        
        # 理想亮度約在 128 (LAB 的 L 範圍是 0-255)
        # 偏離 128 越多，分數越低
        deviation = abs(mean_brightness - 128)
        score = 100 - np.clip(deviation / 128 * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_edge_density(img):
        """
        評估邊緣密度
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 邊緣密度分數 (0-100)
        """
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Canny 邊緣檢測
        edges = cv2.Canny(gray, 50, 150)
        
        # 計算邊緣密度
        edge_density = np.sum(edges > 0) / edges.size
        
        # 正規化到 0-100 (假設好的邊緣密度在 0.05-0.2 範圍)
        score = np.clip(edge_density / 0.2 * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_colorfulness(img):
        """
        評估色彩豐富度 (Hasler and Süsstrunk metric)
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 色彩豐富度分數 (0-100)
        """
        # 分離 RGB 通道
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        # 計算 rg 和 yb
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # 計算標準差和平均值
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        
        # 色彩豐富度指標
        std_rgyb = np.sqrt(std_rg**2 + std_yb**2)
        mean_rgyb = np.sqrt(mean_rg**2 + mean_yb**2)
        
        colorfulness = std_rgyb + 0.3 * mean_rgyb
        
        # 正規化到 0-100 (假設好的色彩豐富度在 0-0.5 範圍)
        score = np.clip(colorfulness / 0.5 * 100, 0, 100)
        
        return score
    
    @staticmethod
    def assess_naturalness(img):
        """
        評估自然度（簡化版）
        
        Args:
            img: RGB影像 (0-1 範圍)
        
        Returns:
            score: 自然度分數 (0-100)
        """
        # 檢查是否有過度飽和或過暗的區域
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        
        # 計算過度飽和的比例
        over_saturated = np.sum(hsv[:, :, 1] > 0.9) / hsv[:, :, 1].size
        
        # 計算過暗和過亮的比例
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        too_dark = np.sum(gray < 0.1) / gray.size
        too_bright = np.sum(gray > 0.9) / gray.size
        
        # 這些比例越低越自然
        unnatural_ratio = over_saturated + too_dark + too_bright
        score = 100 - np.clip(unnatural_ratio * 200, 0, 100)
        
        return score
    
    @staticmethod
    def comprehensive_assessment(img, weights=None):
        """
        綜合品質評估
        
        Args:
            img: RGB影像 (0-1 範圍)
            weights: 各指標權重字典
        
        Returns:
            total_score: 總分 (0-100)
            scores_dict: 各指標分數字典
        """
        # 預設權重
        if weights is None:
            weights = {
                'contrast': 0.20,
                'sharpness': 0.20,
                'entropy': 0.15,
                'saturation': 0.15,
                'brightness': 0.10,
                'edge_density': 0.10,
                'colorfulness': 0.05,
                'naturalness': 0.05
            }
        
        # 計算各項分數
        scores = {}
        
        try:
            scores['contrast'] = QualityAssessment.assess_contrast(img)
        except:
            scores['contrast'] = 50.0
        
        try:
            scores['sharpness'] = QualityAssessment.assess_sharpness(img)
        except:
            scores['sharpness'] = 50.0
        
        try:
            scores['entropy'] = QualityAssessment.assess_entropy(img)
        except:
            scores['entropy'] = 50.0
        
        try:
            scores['saturation'] = QualityAssessment.assess_saturation(img)
        except:
            scores['saturation'] = 50.0
        
        try:
            scores['brightness'] = QualityAssessment.assess_brightness(img)
        except:
            scores['brightness'] = 50.0
        
        try:
            scores['edge_density'] = QualityAssessment.assess_edge_density(img)
        except:
            scores['edge_density'] = 50.0
        
        try:
            scores['colorfulness'] = QualityAssessment.assess_colorfulness(img)
        except:
            scores['colorfulness'] = 50.0
        
        try:
            scores['naturalness'] = QualityAssessment.assess_naturalness(img)
        except:
            scores['naturalness'] = 50.0
        
        # 計算加權總分
        total_score = sum(scores[key] * weights.get(key, 0) for key in scores)
        
        return total_score, scores


def test_quality_assessment():
    """測試品質評估"""
    import matplotlib.pyplot as plt
    
    print("測試品質評估...")
    
    # 創建不同品質的測試影像
    test_images = {
        'Random': np.random.rand(256, 256, 3),
        'Low Contrast': np.ones((256, 256, 3)) * 0.5,
        'High Contrast': np.random.choice([0.0, 1.0], size=(256, 256, 3)),
        'Dark': np.random.rand(256, 256, 3) * 0.3,
        'Bright': 0.7 + np.random.rand(256, 256, 3) * 0.3
    }
    
    # 評估每張影像
    print("\n評估結果:")
    print("-" * 80)
    print(f"{'影像':<15} {'總分':<10} {'對比度':<10} {'清晰度':<10} {'熵':<10} {'飽和度':<10}")
    print("-" * 80)
    
    for name, img in test_images.items():
        total_score, scores = QualityAssessment.comprehensive_assessment(img)
        print(f"{name:<15} {total_score:>6.2f}    "
              f"{scores['contrast']:>6.2f}    "
              f"{scores['sharpness']:>6.2f}    "
              f"{scores['entropy']:>6.2f}    "
              f"{scores['saturation']:>6.2f}")
    
    print("-" * 80)
    print("✓ 品質評估測試完成")


if __name__ == '__main__':
    test_quality_assessment()