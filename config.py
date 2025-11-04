"""
配置文件：定義所有系統參數
"""

import os
from pathlib import Path

class Config:
    """系統配置類別"""
    
    # ===== 路徑設定 =====
    # 輸入影像資料夾（請修改為您的路徑）
    IMAGE_FOLDER = 'D:/rop/UIEBD/raw-890/raw-890'
    
    # 輸出結果資料夾
    OUTPUT_FOLDER = 'D:/rop/results/self_supervised_v1'
    
    # 子資料夾
    FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, 'features')
    STRATEGY_FOLDER = os.path.join(OUTPUT_FOLDER, 'strategy_results')
    MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, 'trained_models')
    REPORT_FOLDER = os.path.join(OUTPUT_FOLDER, 'reports')
    
    # ===== 支援的影像格式 =====
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    # ===== 增強策略參數 =====
    STRATEGIES = {
        'strong_dehazing': {
            'name': 'StrongDehazing',
            'omega': 0.5,
            'guided_radius': 15,
            'L_low': 10,
            'L_high': 95,
            'gamma': 1.2,
            'apply_gamma': True
        },
        'medium_dehazing': {
            'name': 'MediumDehazing',
            'omega': 0.6,
            'guided_radius': 20,
            'L_low': 15,
            'L_high': 92,
            'apply_gamma': True
        },
        'light_enhancement': {
            'name': 'LightEnhancement',
            'omega': 0.4,
            'guided_radius': 10,
            'L_low': 15,
            'L_high': 95,
            'apply_gamma': False
        },
        'clahe_enhancement': {
            'name': 'CLAHEEnhancement',
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8),
            'apply_gamma': False
        },

        # 'weak_dehazing': {
        #     'name': 'WeakDehazing',
        #     'omega': 0.7,
        #     'guided_radius': 25,
        #     'L_low': 20,
        #     'L_high': 90,
        #     'apply_gamma': False
        # },
        
        'histogram_equalization': {
            'name': 'HistogramEqualization',
            'L_low': 10,
            'L_high': 95    
        }
    }
    
    # ===== 品質評估權重 =====
    QUALITY_WEIGHTS = {
        'contrast': 0.25,
        'sharpness': 0.20,
        'entropy': 0.15,
        'saturation': 0.15,
        'brightness': 0.15,
        'edge_density': 0.10
    }
    
    # ===== 特徵提取設定 =====
    # 是否使用深度學習特徵（需要 GPU）
    USE_DEEP_FEATURES = False  # 設為 False 以使用 CPU
    
    # 如果使用深度特徵，選擇模型
    DEEP_FEATURE_MODEL = 'vgg16'  # 可選: 'resnet50', 'efficientnet_b0', 'vgg16'
    
    # ===== 訓練參數 =====
    TEST_SIZE = 0.2  # 測試集比例
    RANDOM_SEED = 42
    CV_FOLDS = 5  # 交叉驗證折數
    
    # ===== 分類器設定 =====
    CLASSIFIERS = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'random_state': RANDOM_SEED
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_SEED
        },
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': RANDOM_SEED
        }
    }
    
    # ===== 其他設定 =====
    # 是否儲存所有策略的增強結果（會佔用較多空間）
    SAVE_ALL_ENHANCED = False
    
    # 是否顯示進度條
    SHOW_PROGRESS = True
    
    # 日誌等級
    LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    @classmethod
    def create_folders(cls):
        """創建所有需要的資料夾"""
        folders = [
            cls.OUTPUT_FOLDER,
            cls.FEATURE_FOLDER,
            cls.STRATEGY_FOLDER,
            cls.MODEL_FOLDER,
            cls.REPORT_FOLDER
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        print("資料夾結構已創建:")
        for folder in folders:
            print(f"  ✓ {folder}")
    
    @classmethod
    def validate(cls):
        """驗證配置"""
        # 檢查輸入資料夾是否存在
        if not os.path.exists(cls.IMAGE_FOLDER):
            print(f"警告: 輸入資料夾不存在: {cls.IMAGE_FOLDER}")
            print("請在 config.py 中修改 IMAGE_FOLDER 路徑")
            return False
        
        # 檢查是否有影像
        image_files = []
        for fmt in cls.SUPPORTED_FORMATS:
            image_files.extend(Path(cls.IMAGE_FOLDER).glob(f'*{fmt}'))
        
        if len(image_files) == 0:
            print(f"警告: 在 {cls.IMAGE_FOLDER} 中找不到支援的影像檔案")
            return False
        
        print(f"✓ 配置驗證通過，找到 {len(image_files)} 張影像")
        return True


# 創建全域配置實例
config = Config()