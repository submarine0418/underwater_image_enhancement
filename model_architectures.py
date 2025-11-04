"""
CNN 版本的參數預測器
直接從影像學習，不需要手動提取特徵
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNParameterPredictor(nn.Module):
    """
    基於 CNN 的參數預測器
    直接輸入影像，輸出參數
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 使用預訓練的 ResNet18 作為特徵提取器
        resnet = models.resnet18(pretrained=pretrained)
        
        # 移除最後的分類層
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 特徵維度 (ResNet18 輸出 512 維)
        feature_dim = 512
        
        # 共享的全連接層
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 參數輸出頭
        self.param_heads = nn.ModuleDict({
            'omega': nn.Linear(128, 1),
            'gamma': nn.Linear(128, 1),
            'L_low': nn.Linear(128, 1),
            'L_high': nn.Linear(128, 1),
            'guided_radius': nn.Linear(128, 1),
            'use_gamma': nn.Linear(128, 1),
        })
        
    def forward(self, x):
        """
        Args:
            x: 影像 tensor (B, 3, H, W)
        """
        # 提取特徵
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # 共享層
        shared = self.shared_layers(features)
        
        # 預測參數
        params = {}
        params['omega'] = torch.sigmoid(self.param_heads['omega'](shared)) * 0.4 + 0.3
        params['gamma'] = torch.sigmoid(self.param_heads['gamma'](shared)) * 0.5 + 1.0
        params['L_low'] = torch.sigmoid(self.param_heads['L_low'](shared)) * 15 + 5
        params['L_high'] = torch.sigmoid(self.param_heads['L_high'](shared)) * 13 + 85
        params['guided_radius'] = torch.sigmoid(self.param_heads['guided_radius'](shared)) * 15 + 10
        params['use_gamma'] = torch.sigmoid(self.param_heads['use_gamma'](shared))
        
        return params


class EfficientNetParameterPredictor(nn.Module):
    """
    基於 EfficientNet 的參數預測器
    更輕量、更快速
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        
        # 載入預訓練的 EfficientNet
        if model_name == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            effnet = efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif model_name == 'efficientnet_b3':
            from torchvision.models import efficientnet_b3
            effnet = efficientnet_b3(pretrained=pretrained)
            feature_dim = 1536
        
        # 移除分類層
        self.feature_extractor = nn.Sequential(*list(effnet.children())[:-1])
        
        # 參數預測層
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 參數頭
        self.param_heads = nn.ModuleDict({
            'omega': nn.Linear(128, 1),
            'gamma': nn.Linear(128, 1),
            'L_low': nn.Linear(128, 1),
            'L_high': nn.Linear(128, 1),
            'guided_radius': nn.Linear(128, 1),
            'use_gamma': nn.Linear(128, 1),
        })
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.predictor(features)
        
        params = {}
        params['omega'] = torch.sigmoid(self.param_heads['omega'](features)) * 0.4 + 0.3
        params['gamma'] = torch.sigmoid(self.param_heads['gamma'](features)) * 0.5 + 1.0
        params['L_low'] = torch.sigmoid(self.param_heads['L_low'](features)) * 15 + 5
        params['L_high'] = torch.sigmoid(self.param_heads['L_high'](features)) * 13 + 85
        params['guided_radius'] = torch.sigmoid(self.param_heads['guided_radius'](features)) * 15 + 10
        params['use_gamma'] = torch.sigmoid(self.param_heads['use_gamma'](features))
        
        return params


class ViTParameterPredictor(nn.Module):
    """
    基於 Vision Transformer 的參數預測器
    最先進的架構
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 使用預訓練的 Vision Transformer
        from torchvision.models import vit_b_16
        vit = vit_b_16(pretrained=pretrained)
        
        # ViT 的特徵維度
        feature_dim = 768
        
        # 移除分類頭
        self.vit_encoder = vit
        self.vit_encoder.heads = nn.Identity()
        
        # 參數預測器
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 參數頭
        self.param_heads = nn.ModuleDict({
            'omega': nn.Linear(128, 1),
            'gamma': nn.Linear(128, 1),
            'L_low': nn.Linear(128, 1),
            'L_high': nn.Linear(128, 1),
            'guided_radius': nn.Linear(128, 1),
            'use_gamma': nn.Linear(128, 1),
        })
    
    def forward(self, x):
        # ViT 編碼
        features = self.vit_encoder(x)
        
        # 預測參數
        features = self.predictor(features)
        
        params = {}
        params['omega'] = torch.sigmoid(self.param_heads['omega'](features)) * 0.4 + 0.3
        params['gamma'] = torch.sigmoid(self.param_heads['gamma'](features)) * 0.5 + 1.0
        params['L_low'] = torch.sigmoid(self.param_heads['L_low'](features)) * 15 + 5
        params['L_high'] = torch.sigmoid(self.param_heads['L_high'](features)) * 13 + 85
        params['guided_radius'] = torch.sigmoid(self.param_heads['guided_radius'](features)) * 15 + 10
        params['use_gamma'] = torch.sigmoid(self.param_heads['use_gamma'](features))
        
        return params


# ============================================
# 使用範例
# ============================================

def create_model(model_type='mlp', pretrained=True):
    """
    創建模型
    
    Args:
        model_type: 'mlp', 'resnet', 'efficientnet', 'vit'
        pretrained: 是否使用預訓練權重
    
    Returns:
        model: 模型實例
    """
    if model_type == 'mlp':
        # 原始 MLP 模型（使用手動特徵）
        from dl_parameter_optimizer import ParameterPredictor
        model = ParameterPredictor(feature_dim=79)
        print("使用 MLP 模型（手動特徵）")
        
    elif model_type == 'resnet':
        model = CNNParameterPredictor(pretrained=pretrained)
        print("使用 ResNet18 模型")
        
    elif model_type == 'efficientnet':
        model = EfficientNetParameterPredictor(model_name='efficientnet_b0', pretrained=pretrained)
        print("使用 EfficientNet-B0 模型")
        
    elif model_type == 'vit':
        model = ViTParameterPredictor(pretrained=pretrained)
        print("使用 Vision Transformer 模型")
        
    else:
        raise ValueError(f"未知的模型類型: {model_type}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    
    return model


if __name__ == '__main__':
    # 測試不同模型
    print("=" * 60)
    print("測試不同模型架構")
    print("=" * 60)
    
    models_to_test = ['mlp', 'resnet', 'efficientnet', 'vit']
    
    for model_type in models_to_test:
        print(f"\n測試 {model_type.upper()} 模型:")
        print("-" * 60)
        
        try:
            model = create_model(model_type, pretrained=True)
            
            # 測試前向傳播
            if model_type == 'mlp':
                # MLP 需要手動特徵
                test_input = torch.randn(2, 79)
            else:
                # CNN/ViT 需要影像
                test_input = torch.randn(2, 3, 224, 224)
            
            with torch.no_grad():
                output = model(test_input)
            
            print("✓ 模型測試通過")
            print(f"輸出參數: {list(output.keys())}")
            
        except Exception as e:
            print(f"✗ 模型測試失敗: {e}")
    
    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60)
