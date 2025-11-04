"""
使用範例和測試腳本
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 確保可以導入專案模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from enhancement_strategies import EnhancementStrategies
from feature_extraction import FeatureExtractor
from quality_assessment import QualityAssessment


def example_1_test_single_strategy():
    """範例 1: 測試單一增強策略"""
    print("=" * 60)
    print("範例 1: 測試單一增強策略")
    print("=" * 60)
    
    # 創建測試影像
    test_img = np.random.rand(256, 256, 3)
    
    # 套用強去霧策略
    params = Config.STRATEGIES['strong_dehazing']
    enhanced = EnhancementStrategies.apply_strategy(test_img, 'strong_dehazing', params)
    
    print(f"✓ 策略 {params['name']} 執行成功")
    print(f"  輸入範圍: [{test_img.min():.3f}, {test_img.max():.3f}]")
    print(f"  輸出範圍: [{enhanced.min():.3f}, {enhanced.max():.3f}]")


def example_2_test_all_strategies():
    """範例 2: 測試所有策略"""
    print("\n" + "=" * 60)
    print("範例 2: 測試所有增強策略")
    print("=" * 60)
    
    test_img = np.random.rand(256, 256, 3)
    
    for strategy_key, params in Config.STRATEGIES.items():
        try:
            enhanced = EnhancementStrategies.apply_strategy(test_img, strategy_key, params)
            print(f"✓ {params['name']:<25} 成功")
        except Exception as e:
            print(f"✗ {params['name']:<25} 失敗: {e}")


def example_3_quality_assessment():
    """範例 3: 品質評估"""
    print("\n" + "=" * 60)
    print("範例 3: 影像品質評估")
    print("=" * 60)
    
    # 創建不同品質的測試影像
    test_images = {
        '隨機影像': np.random.rand(256, 256, 3),
        '低對比度': np.ones((256, 256, 3)) * 0.5,
        '高對比度': np.random.choice([0.0, 1.0], size=(256, 256, 3)),
    }
    
    print(f"\n{'影像類型':<15} {'總分':<10} {'對比度':<10} {'清晰度':<10} {'熵':<10}")
    print("-" * 60)
    
    for name, img in test_images.items():
        total_score, scores = QualityAssessment.comprehensive_assessment(img)
        print(f"{name:<15} {total_score:>6.2f}    "
              f"{scores['contrast']:>6.2f}    "
              f"{scores['sharpness']:>6.2f}    "
              f"{scores['entropy']:>6.2f}")


def example_4_feature_extraction():
    """範例 4: 特徵提取"""
    print("\n" + "=" * 60)
    print("範例 4: 影像特徵提取")
    print("=" * 60)
    
    test_img = np.random.rand(256, 256, 3)
    
    # 提取特徵
    features = FeatureExtractor.extract_all_features(test_img)
    
    print(f"\n特徵統計:")
    print(f"  特徵維度: {len(features)}")
    print(f"  特徵範圍: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  特徵均值: {features.mean():.4f}")
    print(f"  特徵標準差: {features.std():.4f}")
    
    # 檢查特徵品質
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    
    if nan_count == 0 and inf_count == 0:
        print("  ✓ 特徵品質良好（無 NaN 或 Inf）")
    else:
        print(f"  ✗ 警告: 包含 {nan_count} 個 NaN, {inf_count} 個 Inf")


def example_5_compare_strategies():
    """範例 5: 比較所有策略的品質分數"""
    print("\n" + "=" * 60)
    print("範例 5: 比較所有策略")
    print("=" * 60)
    
    # 創建測試影像（模擬霧化影像）
    test_img = np.random.rand(256, 256, 3) * 0.7 + 0.15
    
    print(f"\n{'策略名稱':<25} {'品質分數':<12} {'對比度':<10} {'清晰度':<10}")
    print("-" * 70)
    
    scores_list = []
    
    for strategy_key, params in Config.STRATEGIES.items():
        try:
            # 套用策略
            enhanced = EnhancementStrategies.apply_strategy(test_img, strategy_key, params)
            
            # 評估品質
            total_score, scores = QualityAssessment.comprehensive_assessment(enhanced)
            
            scores_list.append((params['name'], total_score))
            
            print(f"{params['name']:<25} {total_score:>8.2f}    "
                  f"{scores['contrast']:>6.2f}    "
                  f"{scores['sharpness']:>6.2f}")
        except Exception as e:
            print(f"{params['name']:<25} 失敗: {e}")
    
    # 找出最佳策略
    if scores_list:
        best_strategy, best_score = max(scores_list, key=lambda x: x[1])
        print("\n" + "=" * 70)
        print(f"最佳策略: {best_strategy} (分數: {best_score:.2f})")


def example_6_process_real_image():
    """範例 6: 處理真實影像"""
    print("\n" + "=" * 60)
    print("範例 6: 處理真實影像")
    print("=" * 60)
    
    # 檢查是否有測試影像
    if not os.path.exists(Config.IMAGE_FOLDER):
        print(f"影像資料夾不存在: {Config.IMAGE_FOLDER}")
        print("請在 config.py 中設定正確的路徑")
        return
    
    # 尋找第一張影像
    image_files = []
    for fmt in Config.SUPPORTED_FORMATS:
        files = list(Path(Config.IMAGE_FOLDER).glob(f'*{fmt}'))
        if files:
            image_files.extend(files)
            break
    
    if not image_files:
        print(f"在 {Config.IMAGE_FOLDER} 中找不到影像")
        return
    
    # 處理第一張影像
    img_path = image_files[0]
    print(f"\n處理影像: {img_path.name}")
    
    # 讀取影像
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    print(f"影像大小: {img_rgb.shape}")
    
    # 對每個策略評分
    print(f"\n{'策略':<25} {'品質分數':<12}")
    print("-" * 40)
    
    best_strategy = None
    best_score = 0
    
    for strategy_key, params in Config.STRATEGIES.items():
        try:
            enhanced = EnhancementStrategies.apply_strategy(img_rgb, strategy_key, params)
            score, _ = QualityAssessment.comprehensive_assessment(enhanced)
            
            print(f"{params['name']:<25} {score:>8.2f}")
            
            if score > best_score:
                best_score = score
                best_strategy = params['name']
        except Exception as e:
            print(f"{params['name']:<25} 失敗")
    
    print("\n" + "=" * 40)
    print(f"推薦策略: {best_strategy} (分數: {best_score:.2f})")


def example_7_validate_config():
    """範例 7: 驗證配置"""
    print("\n" + "=" * 60)
    print("範例 7: 驗證系統配置")
    print("=" * 60)
    
    print("\n檢查路徑配置:")
    print(f"  輸入資料夾: {Config.IMAGE_FOLDER}")
    print(f"    存在: {'✓' if os.path.exists(Config.IMAGE_FOLDER) else '✗'}")
    
    if os.path.exists(Config.IMAGE_FOLDER):
        # 計算影像數量
        image_count = 0
        for fmt in Config.SUPPORTED_FORMATS:
            image_count += len(list(Path(Config.IMAGE_FOLDER).glob(f'*{fmt}')))
        print(f"    影像數量: {image_count}")
    
    print(f"\n  輸出資料夾: {Config.OUTPUT_FOLDER}")
    
    print("\n檢查策略配置:")
    print(f"  策略數量: {len(Config.STRATEGIES)}")
    for key, params in Config.STRATEGIES.items():
        print(f"    - {params['name']}")
    
    print("\n檢查品質評估權重:")
    total_weight = sum(Config.QUALITY_WEIGHTS.values())
    print(f"  權重總和: {total_weight:.2f} {'✓' if abs(total_weight - 1.0) < 0.01 else '✗ (應為 1.0)'}")
    
    print("\n系統設定:")
    print(f"  使用深度特徵: {Config.USE_DEEP_FEATURES}")
    print(f"  顯示進度條: {Config.SHOW_PROGRESS}")
    print(f"  測試集比例: {Config.TEST_SIZE}")
    print(f"  交叉驗證折數: {Config.CV_FOLDS}")


def run_all_examples():
    """執行所有範例"""
    examples = [
        example_1_test_single_strategy,
        example_2_test_all_strategies,
        example_3_quality_assessment,
        example_4_feature_extraction,
        example_5_compare_strategies,
        example_6_process_real_image,
        example_7_validate_config,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ 範例執行失敗: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有範例執行完成")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='測試和使用範例')
    parser.add_argument('--example', type=int, choices=range(1, 8),
                       help='執行特定範例 (1-7)')
    parser.add_argument('--all', action='store_true',
                       help='執行所有範例')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example:
        example_funcs = [
            example_1_test_single_strategy,
            example_2_test_all_strategies,
            example_3_quality_assessment,
            example_4_feature_extraction,
            example_5_compare_strategies,
            example_6_process_real_image,
            example_7_validate_config,
        ]
        example_funcs[args.example - 1]()
    else:
        print("使用範例腳本")
        print("\n可用的範例:")
        print("  1. 測試單一增強策略")
        print("  2. 測試所有策略")
        print("  3. 品質評估")
        print("  4. 特徵提取")
        print("  5. 比較所有策略")
        print("  6. 處理真實影像")
        print("  7. 驗證配置")
        print("\n使用方式:")
        print("  python example_usage.py --example 1  # 執行範例 1")
        print("  python example_usage.py --all        # 執行所有範例")