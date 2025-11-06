import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from vgg_16_UIE import ImprovedVGGParameterNet, DifferentiableEnhancement,vgg_features


import torchvision.transforms as T

class EnhancementPredictor:
    def __init__(self, model_path, device='cuda', input_size=224):
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.input_size = input_size

        print(f"載入模型: {model_path}  (device={self.device})")
        # instantiate model (avoid pretrained download)
        self.model = ImprovedVGGParameterNet(pretrained=False, hidden_dim=256, use_features=True)
        ckpt = torch.load(model_path, map_location=self.device)
        # checkpoint may contain 'model_state_dict'
        state = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # differentiable enhancement module
        self.enhancer = DifferentiableEnhancement().to(self.device).eval()

        # feature extractor (keeps existing API)
        self.feature_extractor = vgg_features

        # normalization used in training
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        print("✓ 模型與增強模組已載入")

    def _preprocess_for_vgg(self, img):
        # img: H,W,3, RGB float32 [0,1]
        img_resized = cv2.resize((img * 255).astype(np.uint8), (self.input_size, self.input_size),
                                 interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        tensor = self.normalize(tensor)
        return tensor.unsqueeze(0)  # 1,C,H,W

    def _img_to_tensor(self, img):
        # for enhancement module: keep in [0,1]
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return t

    def predict_parameters(self, img):
        # img: H,W,3 RGB [0,1]
        features = self.feature_extractor.extract_all_features(img)  # expected shape (79,)
        feat_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)

        img_vgg = self._preprocess_for_vgg(img).to(self.device)

        with torch.no_grad():
            params_t = self.model(img_vgg, feat_t)
        # convert tensors to python floats
        params = {}
        for k, v in params_t.items():
            # v could be tensor shape (1,1) or (1,)
            val = float(v.cpu().view(-1).item())
            params[k] = val

        # add any extra params expected elsewhere with sensible defaults
        params.setdefault('guided_radius', 15.0)
        params.setdefault('use_gamma', 1.0)

        # clamp for safety (same ranges as training assumptions)
        params['omega'] = float(np.clip(params.get('omega', 0.6), 0.1, 0.9))
        params['gamma'] = float(np.clip(params.get('gamma', 1.2), 0.5, 3.0))
        params['L_low'] = float(np.clip(params.get('L_low', 10.0), 1.0, 30.0))
        params['L_high'] = float(np.clip(params.get('L_high', 90.0), 65.0, 99.0))
        params['guided_radius'] = float(np.clip(params['guided_radius'], 1.0, 50.0))
        params['use_gamma'] = float(np.clip(params['use_gamma'], 0.0, 1.0))

        return params

    def enhance_image(self, img, params=None):
        # img: H,W,3 RGB [0,1]
        if params is None:
            params = self.predict_parameters(img)

        # prepare tensors for enhancement module
        img_t = self._img_to_tensor(img).to(self.device)  # 1,C,H,W

        # build param tensors expected by DifferentiableEnhancement
        # ensure shapes (B,1) for scalars
        param_tensors = {}
        def to_tensor_scalar(x):
            return torch.tensor([ [float(x)] ], dtype=torch.float32, device=self.device)

        param_tensors['omega'] = to_tensor_scalar(params['omega'])
        param_tensors['gamma'] = to_tensor_scalar(params['gamma'])
        param_tensors['L_low'] = to_tensor_scalar(params['L_low'])
        param_tensors['L_high'] = to_tensor_scalar(params['L_high'])

        with torch.no_grad():
            enhanced_t = self.enhancer(img_t, param_tensors)  # returns 1,C,H,W
        enhanced = enhanced_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced, 0.0, 1.0)

        if not np.isfinite(enhanced).all():
            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=1.0, neginf=0.0)
            enhanced = np.clip(enhanced, 0.0, 1.0)

        return enhanced

    def process_single_image(self, input_path, output_path=None, show_params=True):
        input_path = Path(input_path)
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"無法讀取影像: {input_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        params = self.predict_parameters(img)
        if show_params:
            print("\n預測的最佳參數:")
            for k, v in params.items():
                print(f"  {k}: {v:.4f}")

        enhanced = self.enhance_image(img, params)
        enhanced_uint8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_enhanced.png"
        else:
            output_path = Path(output_path)
            if output_path.suffix == '':
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / f"{input_path.stem}_enhanced.png"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)

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

        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 處理: {img_path.name}")
            try:
                out_file = output_path / f"{img_path.stem}_enhanced.png"
                self.process_single_image(str(img_path), str(out_file), show_params=False)
            except Exception as e:
                print(f"  失敗: {e}")

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
    parser = argparse.ArgumentParser(description='使用 vgg-16_UIE 模型進行影像增強')
    parser.add_argument('--input',  type=str, required=True, help='輸入影像或資料夾')
    parser.add_argument('--output', type=str, required=True, help='輸出檔案或資料夾')
    parser.add_argument('--model',  type=str, required=True, help='模型檔案（checkpoint）')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu'])
    args = parser.parse_args()

    predictor = EnhancementPredictor(model_path=args.model, device=args.device)

    input_path = Path(args.input)
    if input_path.is_file():
        predictor.process_single_image(args.input, args.output)
    elif input_path.is_dir():
        predictor.process_folder(args.input, args.output)
    else:
        raise ValueError("輸入路徑不存在或不是檔案/資料夾")
