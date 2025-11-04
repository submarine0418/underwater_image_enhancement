import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json


class ParameterPredictor(nn.Module):
    """
    參數預測網路
    輸入：影像特徵
    輸出：最佳增強參數
    """
    def __init__(self, feature_dim=79, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 輸出各個參數
        self.param_heads = nn.ModuleDict({
            'omega': nn.Linear(hidden_dim // 2, 1),      # omega [0.3-0.7]
            'gamma': nn.Linear(hidden_dim // 2, 1),      # Gamma [1.0-1.5]
            'L_low': nn.Linear(hidden_dim // 2, 1),      # L_low [5-20]
            'L_high': nn.Linear(hidden_dim // 2, 1),     # L_high [85-98]
            'guided_radius': nn.Linear(hidden_dim // 2, 1),  # guided_radius [10-25]
            'use_gamma': nn.Linear(hidden_dim // 2, 1),  # use_gamma [0/1]
        })
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # predict parameters
        params = {}
        params['omega'] = torch.sigmoid(self.param_heads['omega'](features)) * 0.4 + 0.3  # [0.3, 0.7]
        params['gamma'] = torch.sigmoid(self.param_heads['gamma'](features)) * 0.5 + 1.0  # [1.0, 1.5]
        params['L_low'] = torch.sigmoid(self.param_heads['L_low'](features)) * 15 + 5     # [5, 20]
        params['L_high'] = torch.sigmoid(self.param_heads['L_high'](features)) * 13 + 85  # [85, 98]
        params['guided_radius'] = torch.sigmoid(self.param_heads['guided_radius'](features)) * 15 + 10  # [10, 25]
        params['use_gamma'] = torch.sigmoid(self.param_heads['use_gamma'](features))  # [0, 1]
        
        return params


class ReferenceLoss(nn.Module):
    """
    L1  L2 loss
    """
    def __init__(self, l1_weight=0.7, l2_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
    def forward(self, enhanced, reference):
        """
        Args:
            enhanced: 增強後的影像 (B, C, H, W)
            reference: 參考影像 (B, C, H, W)
        """
        l1 = self.l1_loss(enhanced, reference)
        l2 = self.l2_loss(enhanced, reference)
        
        total_loss = self.l1_weight * l1 + self.l2_weight * l2
        
        return total_loss, {'l1': l1.item(), 'l2': l2.item()}


class EnhancementDataset(Dataset):
    """
    增強資料集
    包含輸入影像、參考影像、特徵
    """
    def __init__(self, image_folder, reference_folder, feature_extractor):
        self.image_paths = list(Path(image_folder).glob('*.png')) + \
                          list(Path(image_folder).glob('*.jpg'))
        self.reference_folder = Path(reference_folder)
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 讀取原始影像
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 讀取參考影像
        ref_path = self.reference_folder / img_path.name
        if ref_path.exists():
            ref = cv2.imread(str(ref_path))
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            ref = img.copy()
        
        # 🆕 統一調整大小為 256x256
        target_size = (256, 256)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        ref = cv2.resize(ref, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 提取特徵
        features = self.feature_extractor.extract_all_features(img)
        
        # 轉為 torch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1)
        feature_tensor = torch.from_numpy(features).float()
        
        return {
            'image': img_tensor,
            'reference': ref_tensor,
            'features': feature_tensor,
            'path': str(img_path)
        }

class ParameterOptimizer:
    """
    parameter optimizer trainer
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = ReferenceLoss(l1_weight=0.7, l2_weight=0.3)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        from enhancement_strategies import EnhancementStrategies
        self.enhancer = EnhancementStrategies()
        
    def apply_enhancement_with_params(self, img, params):
        """
        
        Args:
            img: numpy array (H, W, 3), [0, 1]
            params: dict of parameters
        """
        
        param_dict = {
            'use_dehazing': True,
            'omega': float(params['omega']) if not isinstance(params['omega'], float) else params['omega'],
            'gamma': float(params['gamma']) if not isinstance(params['gamma'], float) else params['gamma'],
            'L_low': int(params['L_low']) if not isinstance(params['L_low'], int) else params['L_low'],
            'L_high': int(params['L_high']) if not isinstance(params['L_high'], int) else params['L_high'],
            'guided_radius': int(params['guided_radius']) if not isinstance(params['guided_radius'], int) else params['guided_radius'],
            'use_gamma': float(params['use_gamma']) > 0.5,
            'eps': 5e-1,
            'min_size': 1
                }
        
        # 應用增強
        enhanced = self.enhancer.apply_strategy(img, 'strong_dehazing', param_dict)
        
        return enhanced
    
    def train_epoch(self, dataloader):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        losses = {'l1': 0, 'l2': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features'].to(self.device)
            
            # 預測參數
            predicted_params = self.model(features)
            
            loss = 0
            
            # omega 接近 0.5
            loss += torch.mean((predicted_params['omega'] - 0.5) ** 2)
            
            # gamma 接近 1.2  
            loss += torch.mean((predicted_params['gamma'] - 1.2) ** 2)
            
            # L_low 接近 15
            loss += torch.mean((predicted_params['L_low'] - 15.0) ** 2) * 0.01
            
            # L_high 接近 90
            loss += torch.mean((predicted_params['L_high'] - 90.0) ** 2) * 0.01
            
            # guided_radius 接近 15
            loss += torch.mean((predicted_params['guided_radius'] - 15.0) ** 2) * 0.01
            
            # use_gamma 接近 0.5
            loss += torch.mean((predicted_params['use_gamma'] - 0.5) ** 2)
            
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            losses['l1'] = loss.item()
            losses['l2'] = 0
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        avg_losses = {k: v / len(dataloader) for k, v in losses.items()}
        
        return avg_loss, avg_losses
    
    def validate(self, dataloader):
        """驗證"""
        self.model.eval()
        total_loss = 0
        losses = {'l1': 0, 'l2': 0}
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                references = batch['reference'].to(self.device)
                features = batch['features'].to(self.device)
                
                # 預測並增強
                predicted_params = self.model(features)
                
                batch_size = images.shape[0]
                enhanced_images = []
                
                for i in range(batch_size):
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_params = {k: v[i:i+1] for k, v in predicted_params.items()}
                    enhanced_np = self.apply_enhancement_with_params(img_np, img_params)
                    enhanced_tensor = torch.from_numpy(enhanced_np).permute(2, 0, 1).unsqueeze(0)
                    enhanced_images.append(enhanced_tensor)
                
                enhanced_batch = torch.cat(enhanced_images, dim=0).to(self.device)
                
                # 計算損失
                loss, loss_dict = self.criterion(enhanced_batch, references)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    losses[k] += v
        
        avg_loss = total_loss / len(dataloader)
        avg_losses = {k: v / len(dataloader) for k, v in losses.items()}
        
        return avg_loss, avg_losses
    
    def predict_parameters(self, img_features):
        """predict parameters for a single image"""
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.from_numpy(img_features).float().unsqueeze(0).to(self.device)
            params = self.model(features_tensor)
            
            # 轉為可讀格式
            param_dict = {}
            for k, v in params.items():
                param_dict[k] = float(v.cpu().item())
            
            return param_dict
    
    def save_model(self, path):
        """儲存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        print(f'模型已儲存: {path}')
    
    def load_model(self, path):
        """載入模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f'模型已載入: {path}')


def train_parameter_optimizer(
    image_folder,
    reference_folder,
    output_folder,
    num_epochs=100,
    batch_size=4,
    device='cuda'
):
    """
    訓練參數優化器
    
    Args:
        image_folder: 原始影像資料夾
        reference_folder: 參考影像資料夾
        output_folder: 輸出資料夾
        num_epochs: 訓練週期
        batch_size: 批次大小
        device: 'cuda' 或 'cpu'
    """
    from feature_extraction import FeatureExtractor
    
    print("=" * 60)
    print("deep learnuing parameter optimizer training")
    print("=" * 60)
    
    # 創建輸出資料夾
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 創建資料集
    feature_extractor = FeatureExtractor()
    dataset = EnhancementDataset(image_folder, reference_folder, feature_extractor)
    
    # 分割訓練/驗證集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"訓練集: {train_size} 張")
    print(f"驗證集: {val_size} 張")
    
    # 創建模型
    model = ParameterPredictor(feature_dim=79)
    optimizer_trainer = ParameterOptimizer(model, device=device)
    
    # 訓練歷史
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # 訓練
        train_loss, train_losses = optimizer_trainer.train_epoch(train_loader)
        print(f"訓練 - Loss: {train_loss:.4f}, L1: {train_losses['l1']:.4f}, L2: {train_losses['l2']:.4f}")
        
        # 驗證
        val_loss, val_losses = optimizer_trainer.validate(val_loader)
        print(f"驗證 - Loss: {val_loss:.4f}, L1: {val_losses['l1']:.4f}, L2: {val_losses['l2']:.4f}")
        
        # 記錄
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            optimizer_trainer.save_model(f'{output_folder}/best_model.pth')
            print(f"✓ 最佳模型已更新")
        
        # 定期儲存
        if (epoch + 1) % 10 == 0:
            optimizer_trainer.save_model(f'{output_folder}/model_epoch_{epoch+1}.pth')
    
    # 儲存訓練歷史
    with open(f'{output_folder}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    # 訓練範例
    train_parameter_optimizer(
        image_folder='D:/rop/UIEBD/raw-890/raw-890',
        reference_folder="D:/rop/UIEBD/reference-890/reference-890",
        output_folder='D:/rop/results/dl_optimizer',
        num_epochs=50,
        batch_size=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )