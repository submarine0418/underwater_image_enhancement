"""
影像增強策略實作
包含多種水下影像增強方法
"""

from email.mime import image
import cv2
import numpy as np
from scipy import ndimage
from skimage import color, exposure


class EnhancementStrategies:
    """所有增強策略的集合"""
    
    @staticmethod
    def guided_filter(I, p, r, eps):
        """
        引導濾波
        
        Args:
            I: 引導影像
            p: 輸入影像
            r: 窗口半徑
            eps: 正則化參數
        """
        # 確保輸入是 float64 類型
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
    
    # @staticmethod
    # def estimate_atmospheric_light(img):
    #     """
    #     估計大氣光值（簡化版本）
        
    #     Args:
    #         img: 輸入影像 (0-1 範圍)
        
    #     Returns:
    #         A: 大氣光值 [R, G, B]
    #     """
    #     # 使用最亮的 0.1% 像素
    #     gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    #     flat_gray = gray.flatten()
    #     num_pixels = int(len(flat_gray) * 0.001)  # 0.1%
    #     indices = np.argpartition(flat_gray, -num_pixels)[-num_pixels:]
        
    #     # 在原始彩色影像中找到這些像素
    #     img_flat = img.reshape(-1, 3)
    #     bright_pixels = img_flat[indices]
        
    #     # 取平均作為大氣光
    #     A = np.mean(bright_pixels, axis=0)
    #     A = np.clip(A, 0.1, 1.0)  # 避免極端值
        
    #     return A
    
    @staticmethod

    def estimate_atmospheric_light(image,min_size=1):
        """
        使用四叉樹方法估計大氣光值
        
        參數:
            image: 輸入圖像 (H x W x 3), 值域 [0, 1]
            min_size: 最小區域大小
        
        返回:
            atmosphere: 大氣光值 (H x W x 3)
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
        
        # 將找到的 RGB 值擴展到整個圖像大小
        atmosphere = np.tile(max_RGB.reshape(1, 1, 3), (h, w, 1))
        
        return atmosphere

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


    def get_brightest_pixel(block):
            """
            獲取區域中最亮的像素 RGB 值
            
            參數:
                block: 圖像區域 (H x W x 3)
            
            返回:
                brightest_RGB: 最亮像素的 RGB 值 (3,)
            """
            pixel_sum = np.sum(block, axis=2)
            max_idx = np.argmax(pixel_sum)
            row, col = np.unravel_index(max_idx, pixel_sum.shape)
            brightest_RGB = block[row, col, :]
            
            return brightest_RGB

    @staticmethod
    def estimate_transmission(img, A, omega=0.95, r=15, eps=0.001):
        """
        估計透射圖
        
        Args:
            img: 輸入影像 (0-1 範圍)
            A: 大氣光值
            omega: 去霧參數
            r: 引導濾波半徑
            eps: 引導濾波正則化參數
        """
        # 計算暗通道
        img_norm = img / (A + 1e-10)
        dark_channel = np.min(img_norm, axis=2)
        
        # 初始透射圖
        t_initial = 1 - omega * dark_channel
        
        # 用引導濾波細化
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        t_refined = EnhancementStrategies.guided_filter(gray, t_initial, r, eps)
        
        # 限制範圍
        t_refined = np.clip(t_refined, 0.1, 1.0)
        
        return t_refined
    
    @staticmethod
    def recover_image(img, t, A):
        """
        恢復無霧影像
        
        Args:
            img: 輸入影像
            t: 透射圖
            A: 大氣光值
        """
        t_expanded = np.expand_dims(t, axis=2)
        recovered = (img - A) / t_expanded + A
        recovered = np.clip(recovered, 0, 1)
        return recovered
    
    @staticmethod
    def color_enhancement(img, L_low=15, L_high=95):
        """
        色彩增強（拉伸對比度）
        
        Args:
            img: 輸入影像 (0-1 範圍)
            L_low: 下百分位數
            L_high: 上百分位數
        """
        enhanced = np.zeros_like(img)
        
        for c in range(3):
            channel = img[:, :, c]
            p_low = np.percentile(channel, L_low)
            p_high = np.percentile(channel, L_high)
            
            # 拉伸
            channel_stretched = (channel - p_low) / (p_high - p_low + 1e-10)
            channel_stretched = np.clip(channel_stretched, 0, 1)
            enhanced[:, :, c] = channel_stretched
        
        return enhanced
    
    @staticmethod
    def gamma_correction(img, gamma=1.2):
        """
        Gamma 校正
        
        Args:
            img: 輸入影像 (0-1 範圍)
            gamma: gamma 值
        """
        corrected = np.power(img, 1.0 / gamma)
        return np.clip(corrected, 0, 1)
    
    @staticmethod
    def clahe_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        CLAHE（對比度受限自適應直方圖均衡化）
        
        Args:
            img: 輸入影像 (0-1 範圍)
            clip_limit: 裁剪限制
            tile_grid_size: 網格大小
        """
        # 轉換到 LAB 色彩空間
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        
        # 只對 L 通道做 CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # 轉回 RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float64) / 255.0
    
    # @staticmethod
    # def white_balance(img, percentile=5):
    #     """
    #     簡單白平衡
        
    #     Args:
    #         img: 輸入影像 (0-1 範圍)
    #         percentile: 百分位數
    #     """
    #     balanced = np.zeros_like(img)
        
    #     for c in range(3):
    #         channel = img[:, :, c]
    #         low_val = np.percentile(channel, percentile)
    #         high_val = np.percentile(channel, 100 - percentile)
            
    #         channel_balanced = (channel - low_val) / (high_val - low_val + 1e-10)
    #         balanced[:, :, c] = np.clip(channel_balanced, 0, 1)
        
    #     return balanced
    
    @staticmethod
    def histogram_equalization(img):
        """
        直方圖均衡化
        
        Args:
            img: 輸入影像 (0-1 範圍)
        """
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 分別對每個通道做均衡化
        equalized = np.zeros_like(img_uint8)
        for c in range(3):
            equalized[:, :, c] = cv2.equalizeHist(img_uint8[:, :, c])
        
        return equalized.astype(np.float64) / 255.0
    
    # ===== 完整策略定義 =====
    
    @staticmethod
    def apply_strong_dehazing(img, params):
        """強去霧策略"""
    
        A = EnhancementStrategies.estimate_atmospheric_light(img , min_size=1)
        t = EnhancementStrategies.estimate_transmission(
            img, A, 
            omega=params.get('omega', 0.5),
            r=params.get('guided_radius', 15)
        )
        
        recovered = EnhancementStrategies.recover_image(img, t, A)
     
        enhanced = EnhancementStrategies.color_enhancement(
            recovered,
            L_low=params.get('L_low', 10),
            L_high=params.get('L_high', 95)
        )
        
        if params.get('apply_gamma', False):
            enhanced = EnhancementStrategies.gamma_correction(
                enhanced,
                gamma=params.get('gamma', 1.2)
            )
        
        return enhanced
    
    @staticmethod
    def apply_medium_dehazing(img, params):
        """中度去霧策略"""
        A = EnhancementStrategies.estimate_atmospheric_light(img)
        t = EnhancementStrategies.estimate_transmission(
            img, A,
            omega=params.get('omega', 0.6),
            r=params.get('guided_radius', 20)
        )
        recovered = EnhancementStrategies.recover_image(img, t, A)
        enhanced = EnhancementStrategies.color_enhancement(
            recovered,
            L_low=params.get('L_low', 15),
            L_high=params.get('L_high', 92)
        )
        if params.get('apply_gamma', False):
            enhanced = EnhancementStrategies.gamma_correction(
                enhanced,
                gamma=params.get('gamma', 1.2)
            )
        
        return enhanced
    
    @staticmethod
    def apply_clahe_enhancement(img, params):
        """CLAHE 增強策略"""
        # Step 1: CLAHE
        clahe_result = EnhancementStrategies.clahe_enhancement(
            img,
            clip_limit=params.get('clip_limit', 2.0),
            tile_grid_size=params.get('tile_grid_size', (8, 8))
        )
        
        # Step 2: 色彩增強
        enhanced = EnhancementStrategies.color_enhancement(
            clahe_result,
            L_low=params.get('L_low', 20),
            L_high=params.get('L_high', 85)
        )
        if params.get('apply_gamma', False):
            enhanced = EnhancementStrategies.gamma_correction(
                enhanced,
                gamma=params.get('gamma', 1.2)
            )
        return enhanced
    
    @staticmethod
    def apply_light_enhancement(img, params):
        """輕度增強策略"""
        A = EnhancementStrategies.estimate_atmospheric_light(img)
        t = EnhancementStrategies.estimate_transmission(
            img, A,
            omega=params.get('omega', 0.4),
            r=params.get('guided_radius', 10)
        )
        recovered = EnhancementStrategies.recover_image(img, t, A)
        enhanced = EnhancementStrategies.color_enhancement(
            recovered,
            L_low=params.get('L_low', 15),
            L_high=params.get('L_high', 95)
        )
        if params.get('apply_gamma', False):
            enhanced = EnhancementStrategies.gamma_correction(
                enhanced,
                gamma=params.get('gamma', 1.2)
            )
        
    
        return enhanced
    
    # @staticmethod
    # def apply_white_balance(img, params):
    #     """白平衡策略"""
    #     balanced = EnhancementStrategies.white_balance(
    #         img,
    #         percentile=params.get('percentile', 5)
    #     )
    #     enhanced = EnhancementStrategies.color_enhancement(
    #         balanced,
    #         L_low=params.get('L_low', 15),
    #         L_high=params.get('L_high', 95)
    #     )
    #     return enhanced
    
    @staticmethod
    def apply_histogram_equalization(img, params):
        """直方圖均衡化策略"""
        equalized = EnhancementStrategies.histogram_equalization(img)
        enhanced = EnhancementStrategies.color_enhancement(
            equalized,
            L_low=params.get('L_low', 10),
            L_high=params.get('L_high', 95)
        )
        if params.get('apply_gamma', False):
            enhanced = EnhancementStrategies.gamma_correction(
                enhanced,
                gamma=params.get('gamma', 1.2)
            )
        return enhanced
    
    @staticmethod
    def apply_strategy(img, strategy_name, params):
        """
        統一接口：套用指定策略
        
        Args:
            img: 輸入影像 (RGB, 0-1 範圍)
            strategy_name: 策略名稱
            params: 參數字典
        
        Returns:
            enhanced: 增強後的影像 (RGB, 0-1 範圍)
        """
        strategy_map = {
            'strong_dehazing': EnhancementStrategies.apply_strong_dehazing,
            'medium_dehazing': EnhancementStrategies.apply_medium_dehazing,
            'clahe_enhancement': EnhancementStrategies.apply_clahe_enhancement,
            'light_enhancement': EnhancementStrategies.apply_light_enhancement,
            # 'white_balance': EnhancementStrategies.apply_white_balance,
            'histogram_equalization': EnhancementStrategies.apply_histogram_equalization,
            # 'weak_dehazing': EnhancementStrategies.apply_weak_dehazing
 
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"未知策略: {strategy_name}")
        
        try:
            enhanced = strategy_map[strategy_name](img, params)
            return enhanced
        except Exception as e:
            print(f"策略 {strategy_name} 執行失敗: {e}")
            return img  # 返回原始影像


def test_strategies():
    """測試所有策略"""
    import matplotlib.pyplot as plt
    
    # 創建測試影像
    test_img = np.random.rand(256, 256, 3)
    
    from config import Config
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (strategy_name, params) in enumerate(Config.STRATEGIES.items()):
        print(f"測試策略: {params['name']}")
        try:
            enhanced = EnhancementStrategies.apply_strategy(test_img, strategy_name, params)
            axes[idx].imshow(enhanced)
            axes[idx].set_title(params['name'])
            axes[idx].axis('off')
            print(f"  ✓ 成功")
        except Exception as e:
            print(f"  ✗ 失敗: {e}")
            axes[idx].text(0.5, 0.5, f"Error:\n{str(e)}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('strategy_test.png', dpi=150)
    print("\n測試完成，結果已儲存為 strategy_test.png")


if __name__ == '__main__':
    print("測試增強策略...")
    test_strategies()