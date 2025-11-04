"""
自監督影像增強策略選擇系統 - 主程式
"""

import os
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import Config
from enhancement_strategies import EnhancementStrategies
from feature_extraction import FeatureExtractor
from quality_assessment import QualityAssessment


class SelfSupervisedSystem:
    """自監督影像增強策略選擇系統"""
    
    def __init__(self):
        self.config = Config()
        self.dataset = []
        self.classifier = None
        self.scaler = None
        self.feature_names = []
        
        # 創建資料夾
        self.config.create_folders()
        
        print("=" * 60)
        print("自監督影像增強策略選擇系統")
        print("=" * 60)
    
    def collect_images(self):
        """收集所有影像"""
        image_files = []
        
        for fmt in self.config.SUPPORTED_FORMATS:
            pattern = f'*{fmt}'
            files = list(Path(self.config.IMAGE_FOLDER).glob(pattern))
            image_files.extend(files)
        
        print(f"\n找到 {len(image_files)} 張影像")
        
        if len(image_files) == 0:
            print(f"錯誤: 在 {self.config.IMAGE_FOLDER} 中找不到影像")
            print(f"支援的格式: {self.config.SUPPORTED_FORMATS}")
            return []
        
        return image_files
    
    def build_dataset(self):
        """
        建立自監督資料集
        Phase 1: 對每張影像套用所有策略並評估
        """
        print("\n" + "=" * 60)
        print("Phase 1: 建立自監督資料集")
        print("=" * 60)
        
        # 收集影像
        image_files = self.collect_images()
        if len(image_files) == 0:
            return
        
        # 準備策略
        strategies = self.config.STRATEGIES
        print(f"定義了 {len(strategies)} 種增強策略")
        
        # 創建 CSV 記錄
        csv_path = os.path.join(self.config.REPORT_FOLDER, 'dataset_building.csv')
        csv_data = []
        
        # 處理每張影像
        iterator = tqdm(image_files, desc="處理影像") if self.config.SHOW_PROGRESS else image_files
        
        for img_path in iterator:
            try:
                # 讀取影像
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"警告: 無法讀取 {img_path.name}")
                    continue
                
                # 確保是 3 通道 RGB 影像
                if len(img.shape) == 2:
                    # 灰度影像轉 RGB
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    # RGBA 轉 RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    # BGR 轉 RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 轉換為 float32 (更穩定)
                img_rgb = img.astype(np.float32) / 255.0
                
                # 檢查影像大小
                if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
                    print(f"警告: {img_path.name} 尺寸過小，跳過")
                    continue
                
                # 提取原始特徵
                features = FeatureExtractor.extract_all_features(img_rgb)
                
                # 評估所有策略
                strategy_scores = {}
                enhanced_images = {}
                
                for strategy_key, strategy_params in strategies.items():
                    try:
                        # 套用增強策略
                        enhanced = EnhancementStrategies.apply_strategy(
                            img_rgb, strategy_key, strategy_params
                        )
                        
                        # 評估品質
                        score, _ = QualityAssessment.comprehensive_assessment(
                            enhanced,
                            weights=self.config.QUALITY_WEIGHTS
                        )
                        
                        strategy_scores[strategy_params['name']] = score
                        enhanced_images[strategy_params['name']] = enhanced
                        
                    except Exception as e:
                        if not self.config.SHOW_PROGRESS:
                            print(f"  策略 {strategy_params['name']} 失敗: {e}")
                        strategy_scores[strategy_params['name']] = 0.0
                
                # 選擇最佳策略
                if len(strategy_scores) > 0:
                    best_strategy = max(strategy_scores, key=strategy_scores.get)
                    best_score = strategy_scores[best_strategy]
                    
                    # 儲存最佳結果
                    best_img = enhanced_images[best_strategy]
                    output_path = os.path.join(
                        self.config.STRATEGY_FOLDER,
                        f"{img_path.stem}_{best_strategy}.png"
                    )
                    cv2.imwrite(output_path, 
                               (best_img[:, :, ::-1] * 255).astype(np.uint8))
                    
                    # 儲存到資料集
                    self.dataset.append({
                        'filename': img_path.name,
                        'features': features,
                        'best_strategy': best_strategy,
                        'best_score': best_score,
                        'all_scores': strategy_scores
                    })
                    
                    # 記錄到 CSV
                    csv_row = {
                        'filename': img_path.name,
                        'best_strategy': best_strategy,
                        'best_score': best_score
                    }
                    csv_row.update(strategy_scores)
                    csv_data.append(csv_row)
                
            except Exception as e:
                print(f"處理 {img_path.name} 時發生錯誤: {e}")
                continue
        
        # 儲存 CSV
        if len(csv_data) > 0:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"\n資料集記錄已儲存: {csv_path}")
        
        # 儲存資料集
        if len(self.dataset) > 0:
            dataset_path = os.path.join(self.config.MODEL_FOLDER, 'dataset.pkl')
            with open(dataset_path, 'wb') as f:
                pickle.dump(self.dataset, f)
            print(f"資料集已儲存: {dataset_path}")
            print(f"成功處理 {len(self.dataset)} 張影像")
            
            # 生成統計報告
            self._generate_dataset_report()
        else:
            print("錯誤: 沒有成功處理任何影像")
    
    def _generate_dataset_report(self):
        """生成資料集統計報告"""
        print("\n" + "-" * 60)
        print("資料集統計")
        print("-" * 60)
        
        # 策略分布
        strategies = [item['best_strategy'] for item in self.dataset]
        unique_strategies, counts = np.unique(strategies, return_counts=True)
        
        print("\n策略分布:")
        for strategy, count in zip(unique_strategies, counts):
            percentage = count / len(strategies) * 100
            print(f"  {strategy:<25} {count:>4} ({percentage:>5.1f}%)")
        
        # 品質分數統計
        print("\n平均品質分數:")
        for strategy in unique_strategies:
            scores = [item['best_score'] for item in self.dataset 
                     if item['best_strategy'] == strategy]
            print(f"  {strategy:<25} {np.mean(scores):>6.2f} ± {np.std(scores):>5.2f}")
        
        # 視覺化
        self._visualize_dataset_distribution(unique_strategies, counts)
    
 
    
    def train_classifier(self):
        """
        訓練策略分類器
        Phase 2: 使用自動標註的資料訓練分類器
        """
        print("\n" + "=" * 60)
        print("Phase 2: 訓練策略分類器")
        print("=" * 60)
        
        if len(self.dataset) == 0:
            print("錯誤: 資料集為空，請先執行 build_dataset()")
            return
        
        # 準備訓練資料
        X = np.array([item['features'] for item in self.dataset])
        y = np.array([item['best_strategy'] for item in self.dataset])
        
        print(f"\n訓練資料:")
        print(f"  樣本數: {len(y)}")
        print(f"  特徵維度: {X.shape[1]}")
        print(f"  類別數: {len(np.unique(y))}")
        
        # 檢查類別分布
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"\n類別分布:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  {cls}: {count}")
        
        # 分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_SEED,
            stratify=y
        )
        
        print(f"\n資料分割:")
        print(f"  訓練集: {len(y_train)}")
        print(f"  測試集: {len(y_test)}")
        
        # 標準化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練多個分類器
        classifiers = {
            'RandomForest': RandomForestClassifier(**self.config.CLASSIFIERS['random_forest']),
            'GradientBoosting': GradientBoostingClassifier(**self.config.CLASSIFIERS['gradient_boosting']),
            'SVM': SVC(**self.config.CLASSIFIERS['svm'], probability=True)
        }
        
        results = {}
        
        print("\n訓練分類器:")
        print("-" * 60)
        
        for name, clf in classifiers.items():
            print(f"\n{name}:")
            
            # 訓練
            clf.fit(X_train_scaled, y_train)
            
            # 評估
            train_score = clf.score(X_train_scaled, y_train)
            test_score = clf.score(X_test_scaled, y_test)
            
            # 交叉驗證
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, 
                                       cv=self.config.CV_FOLDS)
            
            results[name] = {
                'model': clf,
                'train_acc': train_score,
                'test_acc': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  訓練準確率: {train_score:.3f}")
            print(f"  測試準確率: {test_score:.3f}")
            print(f"  CV 準確率: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")
        
        # 選擇最佳模型
        best_model_name = max(results, key=lambda x: results[x]['test_acc'])
        self.classifier = results[best_model_name]['model']
        
        print("\n" + "=" * 60)
        print(f"最佳模型: {best_model_name}")
        print(f"測試準確率: {results[best_model_name]['test_acc']:.3f}")
        print("=" * 60)
        
        # 儲存模型
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'best_model_name': best_model_name,
            'results': results,
            'feature_dim': X.shape[1],
            'classes': unique_classes.tolist()
        }
        
        model_path = os.path.join(self.config.MODEL_FOLDER, 'trained_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n模型已儲存: {model_path}")
        
        # 生成詳細報告
        self._generate_classification_report(X_test_scaled, y_test, results)
        
        return results
    
    def _generate_classification_report(self, X_test, y_test, results):
        """生成分類報告"""
        # 使用最佳模型預測
        y_pred = self.classifier.predict(X_test)
        
        # 文字報告
        report_path = os.path.join(self.config.REPORT_FOLDER, 'classification_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("分類器效能報告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 各模型比較
            f.write("模型比較:\n")
            f.write("-" * 70 + "\n")
            for name, res in results.items():
                f.write(f"\n{name}:\n")
                f.write(f"  訓練準確率: {res['train_acc']:.3f}\n")
                f.write(f"  測試準確率: {res['test_acc']:.3f}\n")
                f.write(f"  CV 準確率: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}\n")
            
            # 詳細分類報告
            f.write("\n" + "=" * 70 + "\n")
            f.write("詳細分類報告 (測試集):\n")
            f.write("=" * 70 + "\n\n")
            f.write(classification_report(y_test, y_pred))
            
            # 混淆矩陣
            f.write("\n混淆矩陣:\n")
            cm = confusion_matrix(y_test, y_pred)
            f.write(str(cm))
        
        print(f"分類報告已儲存: {report_path}")
        
        # 視覺化混淆矩陣
        self._plot_confusion_matrix(y_test, y_pred)
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        classes = self.classifier.classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes,
                   yticklabels=classes,
                   cbar_kws={'label': '樣本數'})
        plt.title('混淆矩陣', fontsize=14, pad=20)
        plt.ylabel('真實標籤', fontsize=12)
        plt.xlabel('預測標籤', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.REPORT_FOLDER, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩陣圖已儲存: {plot_path}")
        plt.close()
    
    def predict(self, image_path):
        """
        預測新影像的最佳策略
        
        Args:
            image_path: 影像路徑
        
        Returns:
            prediction: 預測的策略名稱
            probabilities: 各策略的機率
        """
        if self.classifier is None or self.scaler is None:
            raise ValueError("模型尚未訓練，請先執行 train_classifier()")
        
        # 讀取影像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"無法讀取影像: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 提取特徵
        features = FeatureExtractor.extract_all_features(img_rgb)
        features = features.reshape(1, -1)
        
        # 標準化
        features_scaled = self.scaler.transform(features)
        
        # 預測
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # 組合結果
        classes = self.classifier.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        return prediction, prob_dict
    
    def run(self):
        """執行完整流程"""
        print("\n開始執行完整流程...")
        
        # Phase 1: 建立資料集
        self.build_dataset()
        
        if len(self.dataset) == 0:
            print("\n錯誤: 無法建立資料集")
            return
        
        # Phase 2: 訓練分類器
        self.train_classifier()
        
        print("\n" + "=" * 60)
        print("系統訓練完成！")
        print("=" * 60)
        print(f"\n輸出資料夾: {self.config.OUTPUT_FOLDER}")
        print(f"  - 增強結果: {self.config.STRATEGY_FOLDER}")
        print(f"  - 訓練模型: {self.config.MODEL_FOLDER}")
        print(f"  - 分析報告: {self.config.REPORT_FOLDER}")


def main():
    """主程式入口"""
    # 驗證配置
    if not Config.validate():
        print("\n請修改 config.py 中的路徑設定")
        return
    
    # 創建系統
    system = SelfSupervisedSystem()
    
    # 執行完整流程
    system.run()


if __name__ == '__main__':
    main()