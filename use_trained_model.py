import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from deep_learning_parameters import ParameterPredictor, ParameterOptimizer
from feature_extraction import FeatureExtractor
from enhancement_strategies import EnhancementStrategies


class EnhancementPredictor:
    
    def __init__(self, model_path, device='cuda'):
        """
        初始化預測器
        
        Args:
            model_path: results/dl_optimizer/best_model.pth
            device: 'cuda' 或 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 載入模型
        print(f"載入模型: {model_path}")
        self.model = ParameterPredictor(feature_dim=79)
        self.optimizer = ParameterOptimizer(self.model, device=self.device)
        self.optimizer.load_model(model_path)
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor()
        
        # 增強策略
        self.enhancer = EnhancementStrategies()
        
        print(f"✓ 模型已載入（設備: {self.device}）")
    
    def predict_parameters(self, img):
        
        features = self.feature_extractor.extract_all_features(img)
        with torch.no_grad():
            params = self.optimizer.predict_parameters(features)

            # 強制 clamp 參數，防止除以 0 或負數
            params['omega'] = np.clip(params['omega'], 0.1, 0.9)
            params['gamma'] = np.clip(params['gamma'], 0.5, 3.0)
            params['L_low'] = np.clip(params['L_low'], 1.0, 30.0)
            params['L_high'] = np.clip(params['L_high'], 65.0, 99.0)
            params['guided_radius'] = np.clip(params['guided_radius'], 1.0, 50.0)
            params['use_gamma'] = np.clip(params['use_gamma'], 0.0, 1.0)
    
        return params
        
        
    def enhance_image(self, img, params=None):
        """
        增強影像
        
        Args:
            img: numpy array (H, W, 3), RGB, [0, 1]
            params: 參數字典（如果為 None，會自動預測）
        
        Returns:
            enhanced: 增強後的影像 (H, W, 3), RGB, [0, 1]
        """
        # 如果沒有提供參數，先預測
        if params is None:
            params = self.predict_parameters(img)
        
        # 應用增強
        enhanced = self.optimizer.apply_enhancement_with_params(img, params)
        if not np.isfinite(enhanced).all():
            print(f"警告: 偵測到 NaN/inf，強制 clamp 到 [0,1]")
            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0)
            enhanced = np.clip(enhanced, 0.0, 1.0)
        return enhanced
    
    def process_single_image(self, input_path, output_path=None, show_params=True):
        """
        處理單張影像
        """
        input_path = Path(input_path)
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"無法讀取影像: {input_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 預測參數
        params = self.predict_parameters(img)
        if show_params:
            print(f"\n預測的最佳參數:")
            for k, v in params.items():
                print(f"  {k}: {v:.3f}")

        # 增強
        enhanced = self.enhance_image(img, params)
        enhanced = np.clip(enhanced, 0.0, 1.0)  # 再次確保
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        # === 輸出路徑處理 ===
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_enhanced.png"
        else:
            output_path = Path(output_path)
            if output_path.suffix == '':  # 是資料夾
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / f"{input_path.stem}_enhanced.png"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)

        # === 儲存 ===
        enhanced_uint8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), enhanced_bgr)
        print(f"儲存: {output_path}")
        
        return enhanced, params
    
    def process_folder(self, input_folder, output_folder):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.png')) + \
                    list(input_path.glob('*.jpg')) + \
                    list(input_path.glob('*.jpeg'))
        
        if not image_files:
            print("找不到任何影像檔案！")
            return

        print(f"\n找到 {len(image_files)} 張影像")
        print(f"開始批量處理...\n")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 處理: {img_path.name}")
            try:
                output_file = output_path / f"{img_path.stem}_enhanced.png"
                self.process_single_image(str(img_path), str(output_file), show_params=False)
            except Exception as e:
                print(f"  失敗: {e}")
        
        print(f"\n批量處理完成！")
        print(f"輸出資料夾: {output_path}")

# ============================================
# 使用範例
# ============================================

def example_single_image():
    """範例 1: 處理單張影像"""
    print("=" * 60)
    print("範例 1: 處理單張影像")
    print("=" * 60)
    
    # 載入模型
    predictor = EnhancementPredictor(
        model_path='D:/rop/results/dl_optimizer/best_model.pth',
        device='cuda'
    )
    
    # 處理單張影像
    predictor.process_single_image(
        input_path='D:/rop/UIEBD/raw-890/100_img.png',
        output_path='D:/rop/results/test_enhanced.png'
    )


def example_batch_processing():
    """範例 2: 批量處理"""
    print("=" * 60)
    print("範例 2: 批量處理")
    print("=" * 60)
    
    # 載入模型
    predictor = EnhancementPredictor(
        model_path='D:/rop/results/dl_optimizer/best_model.pth',
        device='cuda'
    )
    
    # 批量處理
    predictor.process_folder(
        input_folder='D:/rop/UIEBD/challenging-60/challenging-60',
        output_folder='D:/rop/results/enhanced_output'
    )


def example_custom_enhancement():
    """範例 3: 自訂參數增強"""
    print("=" * 60)
    print("範例 3: 使用自訂參數")
    print("=" * 60)
    
    # 載入模型
    predictor = EnhancementPredictor(
        model_path='D:/rop/results/dl_optimizer/best_model.pth'
    )
    
    # 讀取影像
    img = cv2.imread("D:/rop/UIEBD/raw-890/raw-890/8_img_.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 方案 A: 使用模型預測的參數
    params_predicted = predictor.predict_parameters(img)
    enhanced_auto = predictor.enhance_image(img, params_predicted)
  
    
    # 儲存
    cv2.imwrite('enhanced_auto.png', 
                cv2.cvtColor((enhanced_auto * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
  
    
  


def example_compare_results():
    """範例 4: 比較原始、預測、增強"""
    print("=" * 60)
    print("範例 4: 視覺化比較")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    # 載入模型
    predictor = EnhancementPredictor(
        model_path='D:/rop/results/dl_optimizer/best_model.pth'
    )
    
    # 讀取影像
    img = cv2.imread('D:/rop/UIEBD/raw-890/raw-890/8_img_.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 增強
    enhanced, params = predictor.process_single_image(
        'D:/rop/UIEBD/raw-890/raw-890/8_img_.png',
        show_params=True
    )
    
    # 視覺化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img)
    axes[0].set_title('原始影像')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title('增強影像')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('D:/rop/results/comparison.png', dpi=150)
    print("✓ 比較圖已儲存")


# ============================================
# 主程式
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用訓練好的模型進行影像增強')
    parser.add_argument('--input',  type=str, required=True, help='輸入影像或資料夾')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾')
    parser.add_argument('--model',  type=str, required=True, help='模型路徑')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu'])

    args = parser.parse_args()

    predictor = EnhancementPredictor(model_path=args.model, device=args.device)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # 自動判斷：單張 vs 資料夾
    if input_path.is_file():                     # 單張圖片
        predictor.process_single_image(args.input, args.output)
    elif input_path.is_dir():                    # 資料夾
        predictor.process_folder(args.input, args.output)
    else:
        raise ValueError("輸入路徑不存在或不是檔案/資料夾")
