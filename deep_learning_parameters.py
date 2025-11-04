"""
端到端深度學習參數優化器
直接優化 L1+L2 Loss（增強結果 vs 參考影像）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm


# ============================================
# 可微分的影像增強操作（PyTorch 版本）
# ============================================

class DifferentiableEnhancement(nn.Module):
    """
    可微分的影像增強
    所有操作都用 PyTorch，保持梯度流
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, img, params):
        """
        Args:
            img: (B, C, H, W) tensor, [0, 1]
            params: 參數字典
        
        Returns:
            enhanced: (B, C, H, W) tensor, [0, 1]
        """
        B, C, H, W = img.shape
        
        # 1. 色彩拉伸（可微分）
        enhanced = self.color_stretch(img, params['L_low'], params['L_high'])
        
        # 2. Gamma 校正（可微分）
        use_gamma = params['use_gamma'].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        gamma_val = params['gamma'].view(-1, 1, 1, 1)
        
        gamma_enhanced = torch.pow(enhanced + 1e-8, 1.0 / gamma_val)
        enhanced = use_gamma * gamma_enhanced + (1 - use_gamma) * enhanced
        
        # 確保範圍
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced
    
    def color_stretch(self, img, L_low, L_high):
        """
        可微分的色彩拉伸
        
        Args:
            img: (B, C, H, W)
            L_low: (B, 1) 下百分位
            L_high: (B, 1) 上百分位
        """
        B, C, H, W = img.shape
        
        # 對每個 batch 和 channel 計算百分位數（近似）
        enhanced = torch.zeros_like(img)
        
        for b in range(B):
            for c in range(C):
                channel = img[b, c]
                
                # 使用可微分的百分位數近似
                sorted_vals, _ = torch.sort(channel.flatten())
                n = sorted_vals.shape[0]
                
                low_idx = int(L_low[b].item() / 100.0 * n)
                high_idx = int(L_high[b].item() / 100.0 * n)
                
                p_low = sorted_vals[low_idx]
                p_high = sorted_vals[high_idx]
                
                # 拉伸
                stretched = (channel - p_low) / (p_high - p_low + 1e-8)
                enhanced[b, c] = torch.clamp(stretched, 0, 1)
        
        return enhanced


# ============================================
# 模型架構（帶殘差連接）
# ============================================

class ResidualBlock(nn.Module):
    """殘差塊"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.relu(self.dropout(self.block(x) + x))


class ParameterPredictor(nn.Module):
    """
    參數預測網路
    輸入：影像特徵 (79維)
    輸出：增強參數
    """
    def __init__(self, feature_dim=79, hidden_dim=256, num_blocks=3):
        super().__init__()
        
        # 輸入投影
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 殘差塊
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 輸出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 參數頭
        self.param_heads = nn.ModuleDict({
            'gamma': nn.Linear(hidden_dim // 2, 1),
            'L_low': nn.Linear(hidden_dim // 2, 1),
            'L_high': nn.Linear(hidden_dim // 2, 1),
            'use_gamma': nn.Linear(hidden_dim // 2, 1),
        })
        
    def forward(self, x):
        # 特徵提取
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        features = self.output_proj(x)
        
        # 預測參數（帶梯度）
        params = {}
        params['gamma'] = torch.sigmoid(self.param_heads['gamma'](features)) * 0.5 + 1.0  # [1.0, 1.5]
        params['L_low'] = torch.sigmoid(self.param_heads['L_low'](features)) * 15 + 5      # [5, 20]
        params['L_high'] = torch.sigmoid(self.param_heads['L_high'](features)) * 13 + 85   # [85, 98]
        params['use_gamma'] = torch.sigmoid(self.param_heads['use_gamma'](features))       # [0, 1]
        
        return params


# ============================================
# 損失函數
# ============================================

class ReferenceLoss(nn.Module):
    """
    參考影像損失：L1 + L2
    """
    def __init__(self, l1_weight=0.5, l2_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
    def forward(self, enhanced, reference):
        """
        Args:
            enhanced: (B, C, H, W) 增強後的影像
            reference: (B, C, H, W) 參考影像
        """
        l1 = self.l1_loss(enhanced, reference)
        l2 = self.l2_loss(enhanced, reference)
        
        total_loss = self.l1_weight * l1 + self.l2_weight * l2
        
        return total_loss, {'l1': l1.item(), 'l2': l2.item()}


# ============================================
# 資料集
# ============================================

class EnhancementDataset(Dataset):
    """
    增強資料集
    包含：原始影像、參考影像、特徵
    """
    def __init__(self, image_folder, reference_folder, feature_extractor, target_size=256):
        self.image_paths = list(Path(image_folder).glob('*.png')) + \
                          list(Path(image_folder).glob('*.jpg'))
        self.reference_folder = Path(reference_folder)
        self.feature_extractor = feature_extractor
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 讀取原始影像
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))
        img = img.astype(np.float32) / 255.0
        
        # 讀取參考影像
        ref_path = self.reference_folder / img_path.name
        if ref_path.exists():
            ref = cv2.imread(str(ref_path))
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            ref = cv2.resize(ref, (self.target_size, self.target_size))
            ref = ref.astype(np.float32) / 255.0
        else:
            ref = img.copy()
        
        # 提取特徵
        features = self.feature_extractor.extract_all_features(img)
        
        # 轉為 tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # (C, H, W)
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).float()
        feature_tensor = torch.from_numpy(features).float()
        
        return {
            'image': img_tensor,
            'reference': ref_tensor,
            'features': feature_tensor,
            'path': str(img_path)
        }


# ============================================
# 訓練器
# ============================================

class EndToEndTrainer:
    """
    端到端訓練器
    直接優化 L1+L2 Loss
    """
    def __init__(self, param_predictor, device='cuda'):
        self.param_predictor = param_predictor.to(device)
        self.enhancement = DifferentiableEnhancement().to(device)
        self.criterion = ReferenceLoss(l1_weight=0.5, l2_weight=0.5)
        self.optimizer = optim.Adam(self.param_predictor.parameters(), lr=1e-4)
        self.device = device
        
    def train_epoch(self, dataloader):
        """訓練一個 epoch"""
        self.param_predictor.train()
        total_loss = 0
        losses = {'l1': 0, 'l2': 0}
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)         # (B, C, H, W)
            references = batch['reference'].to(self.device)
            features = batch['features'].to(self.device)    # (B, 79)
            
            # 1. 預測參數（保持梯度）
            predicted_params = self.param_predictor(features)
            
            # 2. 應用可微分的增強
            enhanced = self.enhancement(images, predicted_params)
            
            # 3. 計算損失
            loss, loss_dict = self.criterion(enhanced, references)
            
            # 4. 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止爆炸）
            torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 記錄
            total_loss += loss.item()
            for k, v in loss_dict.items():
                losses[k] += v
            
            # 更新進度條
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        avg_losses = {k: v / len(dataloader) for k, v in losses.items()}
        
        return avg_loss, avg_losses
    
    def validate(self, dataloader):
        """驗證"""
        self.param_predictor.eval()
        total_loss = 0
        losses = {'l1': 0, 'l2': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                references = batch['reference'].to(self.device)
                features = batch['features'].to(self.device)
                
                # 預測並增強
                predicted_params = self.param_predictor(features)
                enhanced = self.enhancement(images, predicted_params)
                
                # 計算損失
                loss, loss_dict = self.criterion(enhanced, references)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    losses[k] += v
        
        avg_loss = total_loss / len(dataloader)
        avg_losses = {k: v / len(dataloader) for k, v in losses.items()}
        
        return avg_loss, avg_losses
    
    def save_model(self, path):
        """儲存模型"""
        torch.save({
            'param_predictor': self.param_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f'模型已儲存: {path}')
    
    def load_model(self, path):
        """載入模型"""
        checkpoint = torch.load(path, weights_only=False)
        self.param_predictor.load_state_dict(checkpoint['param_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'模型已載入: {path}')


# ============================================
# 訓練主函數
# ============================================

def train_end_to_end(
    image_folder,
    reference_folder,
    output_folder,
    num_epochs=10,
    batch_size=4,
    device='cuda'
):
    """
    端到端訓練
    
    Args:
        image_folder: 原始影像資料夾
        reference_folder: 參考影像資料夾（高品質目標）
        output_folder: 輸出資料夾
        num_epochs: 訓練週期
        batch_size: 批次大小
        device: 'cuda' 或 'cpu'
    """
    from feature_extraction import FeatureExtractor
    
    print("=" * 60)
    print("端到端深度學習參數優化（L1+L2 Loss）")
    print("=" * 60)
    
    # 創建輸出資料夾
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 檢查 CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'
    
    # 創建資料集
    print("\n載入資料集...")
    feature_extractor = FeatureExtractor()
    dataset = EnhancementDataset(image_folder, reference_folder, feature_extractor, target_size=256)
    
    if len(dataset) == 0:
        print(f"錯誤：找不到影像！")
        print(f"  原始影像: {image_folder}")
        print(f"  參考影像: {reference_folder}")
        return
    
    # 分割訓練/驗證集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"訓練集: {train_size} 張")
    print(f"驗證集: {val_size} 張")
    print(f"設備: {device}")
    
    # 創建模型和訓練器
    print("\n創建模型...")
    model = ParameterPredictor(feature_dim=79, hidden_dim=256, num_blocks=3)
    trainer = EndToEndTrainer(model, device=device)
    
    # 訓練歷史
    history = {'train_loss': [], 'val_loss': [], 'train_l1': [], 'train_l2': []}
    best_val_loss = float('inf')
    
    # 訓練循環
    print("\n開始訓練...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # 訓練
        train_loss, train_losses = trainer.train_epoch(train_loader)
        print(f"訓練 - Loss: {train_loss:.4f}, L1: {train_losses['l1']:.4f}, L2: {train_losses['l2']:.4f}")
        
        # 驗證
        val_loss, val_losses = trainer.validate(val_loader)
        print(f"驗證 - Loss: {val_loss:.4f}, L1: {val_losses['l1']:.4f}, L2: {val_losses['l2']:.4f}")
        
        # 記錄
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_l1'].append(train_losses['l1'])
        history['train_l2'].append(train_losses['l2'])
        
        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(f'{output_folder}/best_model.pth')
            print(f"✓ 最佳模型已更新")
        
        # 定期儲存
        if (epoch + 1) % 10 == 0:
            trainer.save_model(f'{output_folder}/model_epoch_{epoch+1}.pth')
    
    # 儲存訓練歷史
    with open(f'{output_folder}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"最佳驗證損失: {best_val_loss:.4f}")
    print(f"模型已儲存: {output_folder}/best_model.pth")
    print("=" * 60)


# ============================================
# 主程式
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='端到端參數優化訓練')
    parser.add_argument('--input', type=str, default='D:/rop/UIEBD/raw-890/raw-890',
                       help='原始影像資料夾')
    parser.add_argument('--reference', type=str, default='D:/rop/UIEBD/reference-890/reference-890',
                       help='參考影像資料夾')
    parser.add_argument('--output', type=str, default='D:/rop/results/e2e_optimizer',
                       help='輸出資料夾')
    parser.add_argument('--epochs', type=int, default=50, help='訓練週期')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
    
    args = parser.parse_args()
    
    train_end_to_end(
        image_folder=args.input,
        reference_folder=args.reference,
        output_folder=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )