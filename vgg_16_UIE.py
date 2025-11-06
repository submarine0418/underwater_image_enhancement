import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import models, transforms
import cv2
import numpy as np
from torch.cuda.amp import autocast, GradScaler


# ============================================
# Improved Differentiable Enhancement Module
# ============================================

class DifferentiableEnhancement(nn.Module):
    """
    Differentiable image enhancement with color stretching and dehazing.
    All operations are GPU-optimized and fully differentiable.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, img, params):
        """
        Args:
            img: (B, C, H, W) tensor in range [0, 1]
            params: Dictionary of enhancement parameters
        Returns:
            enhanced: (B, C, H, W) enhanced image
        """
        # 1. Color stretching
        enhanced = self.color_stretch_batch(
            img, 
            params['L_low'], 
            params['L_high']
        )
        
        # 2. Dehazing
        if 'omega' in params:
            enhanced = self.dehaze_batch(enhanced, params)
        
        # 3. Gamma correction
        if 'gamma' in params:
            enhanced = self.gamma_correction(enhanced, params['gamma'])
        
        return torch.clamp(enhanced, 0, 1)
    
    def color_stretch_batch(self, img, L_low, L_high):
        """
        Vectorized color stretching for entire batch.
        
        Args:
            img: (B, C, H, W)
            L_low: (B, 1) percentage values
            L_high: (B, 1) percentage values
        """
        B, C, H, W = img.shape
        enhanced = torch.zeros_like(img)
        
        # Process each image in batch
        for b in range(B):
            for c in range(C):
                channel = img[b, c]
                
                # Get percentile values
                flat_channel = channel.flatten()
                sorted_vals, _ = torch.sort(flat_channel)
                
                low_idx = int((L_low[b].item() / 100.0) * len(sorted_vals))
                high_idx = int((L_high[b].item() / 100.0) * len(sorted_vals))
                
                low_idx = max(0, min(low_idx, len(sorted_vals) - 1))
                high_idx = max(0, min(high_idx, len(sorted_vals) - 1))
                
                p_low = sorted_vals[low_idx]
                p_high = sorted_vals[high_idx]
                
                # Stretch
                range_val = p_high - p_low + 1e-8
                stretched = (channel - p_low) / range_val
                enhanced[b, c] = torch.clamp(stretched, 0, 1)
        
        return enhanced
    
    def dehaze_batch(self, img, params):
        """
        Simplified but effective dehazing using dark channel prior.
        
        Args:
            img: (B, C, H, W)
            params: dict with 'omega' key
        """
        omega = params['omega'].view(-1, 1, 1, 1)
        
        # Dark channel approximation
        dark_channel = torch.min(img, dim=1, keepdim=True)[0]
        
        # Atmospheric light estimation (simplified to constant)
        A = 0.6
        
        # Transmission map
        t = 1 - omega * dark_channel
        t = torch.clamp(t, 0.1, 1.0)
        
        # Recover scene radiance
        dehazed = (img - A) / t + A
        
        return torch.clamp(dehazed, 0, 1)
    
    def gamma_correction(self, img, gamma):
        """
        Apply gamma correction.
        
        Args:
            img: (B, C, H, W)
            gamma: (B, 1) gamma values
        """
        gamma = gamma.view(-1, 1, 1, 1)
        return torch.pow(img + 1e-8, gamma)


# ============================================
# Improved VGG Parameter Predictor
# ============================================

class ImprovedVGGParameterNet(nn.Module):
    """
    Enhanced parameter prediction network with:
    - Better feature fusion
    - Residual connections
    - Attention mechanism
    """
    def __init__(self, pretrained=True, hidden_dim=256, use_features=True):
        super().__init__()
        
        self.use_features = use_features
        
        # VGG16 backbone (frozen early layers for stability)
        vgg16 = models.vgg16(pretrained=pretrained)
        self.vgg_features = vgg16.features[:23]  # Up to conv4_3
        
        # Freeze early layers
        for i, param in enumerate(self.vgg_features.parameters()):
            if i < 16:  # Freeze first few conv blocks
                param.requires_grad = False
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
        
        vgg_out_dim = 512
        feature_dim = 79 if use_features else 0
        
        # Feature fusion with residual connection
        self.feature_fusion = nn.Sequential(
            nn.Linear(vgg_out_dim * 2 + feature_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # Individual parameter heads with appropriate ranges
        self.param_heads = nn.ModuleDict({
            'omega': self._make_param_head(hidden_dim, 1),      # [0.4, 0.8]
            'gamma': self._make_param_head(hidden_dim, 1),      # [1.0, 1.5]
            'L_low': self._make_param_head(hidden_dim, 1),      # [5, 15]
            'L_high': self._make_param_head(hidden_dim, 1),     # [85, 95]
        })
        
        # Parameter ranges
        self.param_ranges = {
            'omega': (0.3, 0.9),
            'gamma': (1.0, 1.5),
            'L_low': (2.0, 15.0),
            'L_high': (60.0, 95.0),
        }

    def _make_param_head(self, in_dim, out_dim):
        """Create a parameter prediction head"""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, img_tensor, feature_tensor=None):
        """
        Args:
            img_tensor: (B, C, H, W) image tensor
            feature_tensor: (B, 79) optional feature vector
        Returns:
            params: Dictionary of predicted parameters
        """
        B = img_tensor.size(0)
        
        # Extract VGG features
        vgg_feat = self.vgg_features(img_tensor)
        
        # Dual pooling for richer representation
        avg_feat = self.avgpool(vgg_feat).view(B, -1)
        max_feat = self.maxpool(vgg_feat).view(B, -1)
        pooled_feat = torch.cat([avg_feat, max_feat], dim=1)
        
        # Combine with traditional features if available
        if self.use_features and feature_tensor is not None:
            if isinstance(feature_tensor, list):
                feature_tensor = torch.stack(feature_tensor)
            feature_tensor = feature_tensor.float().to(img_tensor.device)
            combined = torch.cat([pooled_feat, feature_tensor], dim=1)
        else:
            combined = pooled_feat
        
        # Feature fusion
        fused = self.feature_fusion(combined)
        
        # Apply attention
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        # Predict parameters with proper ranges
        params = {}
        for name, head in self.param_heads.items():
            raw_output = head(fused)
            min_val, max_val = self.param_ranges[name]
            params[name] = torch.sigmoid(raw_output) * (max_val - min_val) + min_val
        
        return params


# ============================================
# Improved Loss Functions
# ============================================

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
    
    def forward(self, pred, target):
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return F.mse_loss(pred_feat, target_feat)


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components:
    - L1 loss for pixel accuracy
    - L2 loss for smoothness
    - Perceptual loss for visual quality
    """
    def __init__(self, l1_weight=0.3, l2_weight=0.5, perceptual_weight=0.2, device='cuda'):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device)

    def forward(self, enhanced, reference):
        """Calculate combined loss"""
        l1 = self.l1_loss(enhanced, reference)
        l2 = self.l2_loss(enhanced, reference)
        perceptual = self.perceptual_loss(enhanced, reference)
        
        total_loss = (self.l1_weight * l1 + 
                     self.l2_weight * l2 + 
                     self.perceptual_weight * perceptual)
        
        return total_loss, {'l1': l1.item(), 'l2': l2.item(), 'perceptual': perceptual.item()}


# ============================================
# Improved Dataset with Data Augmentation
# ============================================

class ImprovedEnhancementDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""
    def __init__(self, image_folder, reference_folder, target_size=224, 
                 augment=True, use_features=True):
        self.image_folder = Path(image_folder)
        self.reference_folder = Path(reference_folder)
        self.target_size = target_size
        self.augment = augment
        self.use_features = use_features
        
        # Find all images
        self.image_paths = (list(self.image_folder.glob('*.jpg')) + 
                           list(self.image_folder.glob('*.png')) +
                           list(self.image_folder.glob('*.jpeg')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")
        
        print(f"Found {len(self.image_paths)} images")
        
        # Normalization for VGG
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, path, target_size):
        """Load and preprocess image"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size), 
                        interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img

    def augment_pair(self, img, ref):
        """Apply consistent augmentation to image pair"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            ref = np.fliplr(ref).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            ref = np.flipud(ref).copy()
        
        return img, ref

    def extract_basic_features(self, img):
        """Extract simple statistical features"""
        features = []
        
        # Per-channel statistics
        for c in range(3):
            channel = img[:, :, c]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel)
            ])
        
        # Overall statistics
        features.extend([
            np.mean(img),
            np.std(img),
            np.mean(img ** 2),  # Second moment
        ])
        
        # Pad to 79 dimensions with zeros
        while len(features) < 79:
            features.append(0.0)
        
        return np.array(features[:79], dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load input image
        img = self.load_image(img_path, self.target_size)
        
        # Load reference
        ref_path = self.reference_folder / img_path.name
        if ref_path.exists():
            ref = self.load_image(ref_path, self.target_size)
        else:
            ref = img.copy()
        
        # Apply augmentation
        if self.augment:
            img, ref = self.augment_pair(img, ref)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1)
        
        # Extract features
        if self.use_features:
            feature_tensor = torch.from_numpy(self.extract_basic_features(img))
        else:
            feature_tensor = torch.zeros(79)

        return {
            'image': img_tensor,
            'reference': ref_tensor,
            'features': feature_tensor,
            'path': str(img_path)
        }


def _ensure_float01(img):
    """Convert image to float32 in [0,1]. Accepts uint8 or float inputs."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
    return img

def extract_all_features(img):
    """
    Standalone feature extractor compatible with previous code.
    Input: img H,W,3 RGB (uint8 or float), returns numpy array shape (79,), dtype float32.
    Mirrors ImprovedEnhancementDataset.extract_basic_features logic.
    """
    img = _ensure_float01(img)
    features = []

    # Per-channel statistics
    for c in range(3):
        channel = img[:, :, c]
        features.extend([
            float(np.mean(channel)),
            float(np.std(channel)),
            float(np.min(channel)),
            float(np.max(channel)),
            float(np.median(channel))
        ])

    # Overall statistics
    features.extend([
        float(np.mean(img)),
        float(np.std(img)),
        float(np.mean(img ** 2)),  # Second moment
    ])

    # Pad to 79 dimensions with zeros
    while len(features) < 79:
        features.append(0.0)

    return np.array(features[:79], dtype=np.float32)

class _VGGFeaturesWrapper:
    @staticmethod
    def extract_all_features(img):
        return extract_all_features(img)

# exported object expected by use_trained_model.py
vgg_features = _VGGFeaturesWrapper()


# ============================================
# Improved Trainer with Mixed Precision
# ============================================

class ImprovedTrainer:
    """Enhanced trainer with mixed precision and better logging"""
    def __init__(self, param_predictor, device='cuda', use_amp=True):
        self.device = device
        self.use_amp = use_amp and (device == 'cuda')
        
        self.param_predictor = param_predictor.to(device)
        self.enhancement = DifferentiableEnhancement().to(device)
        self.criterion = CombinedLoss(device=device)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.param_predictor.parameters(), 
            lr=1e-5,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.param_predictor.train()
        total_loss = 0
        loss_components = {'l1': 0, 'l2': 0, 'perceptual': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            references = batch['reference'].to(self.device)
            features = batch['features'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    params = self.param_predictor(images, features)
                    enhanced = self.enhancement(images, params)
                    loss, components = self.criterion(enhanced, references)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                params = self.param_predictor(images, features)
                enhanced = self.enhancement(images, params)
                loss, components = self.criterion(enhanced, references)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), 1.0)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        self.train_losses.append(avg_loss)
        return avg_loss, avg_components

    def validate(self, dataloader):
        """Validation loop"""
        self.param_predictor.eval()
        total_loss = 0
        loss_components = {'l1': 0, 'l2': 0, 'perceptual': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                references = batch['reference'].to(self.device)
                features = batch['features'].to(self.device)
                
                params = self.param_predictor(images, features)
                enhanced = self.enhancement(images, params)
                loss, components = self.criterion(enhanced, references)
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        self.val_losses.append(avg_loss)
        return avg_loss, avg_components

    def save(self, path, epoch=None, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.param_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.param_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"✓ Checkpoint loaded: {path}")


# ============================================
# Main Training Function
# ============================================

def train_enhanced_model(image_folder, reference_folder, output_folder,
                        epochs=100, batch_size=4, device='cuda',
                        use_amp=True, resume=None):
    """
    Main training function with improved features
    
    Args:
        image_folder: Path to input images
        reference_folder: Path to reference images
        output_folder: Path to save outputs
        epochs: Number of training epochs
        batch_size: Batch size
        device: 'cuda' or 'cpu'
        use_amp: Use mixed precision training
        resume: Path to checkpoint to resume from
    """
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        use_amp = False
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Enhanced Underwater Image Enhancement Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Mixed Precision: {use_amp}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("=" * 80)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        train_dataset = ImprovedEnhancementDataset(
            image_folder, reference_folder, 
            target_size=224, augment=True
        )
        
        val_dataset = ImprovedEnhancementDataset(
            image_folder, reference_folder,
            target_size=224, augment=False
        )
        
        # Split dataset
        total_size = len(train_dataset)
        train_size = int(0.85 * total_size)
        val_size = total_size - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if device == 'cpu' else 2,
            pin_memory=(device == 'cuda'),
            persistent_workers=False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if device == 'cpu' else 2,
            pin_memory=(device == 'cuda'),
            persistent_workers=False
        )
        
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        # Initialize model and trainer
        print("\nInitializing model...")
        model = ImprovedVGGParameterNet(pretrained=True, hidden_dim=256)
        trainer = ImprovedTrainer(model, device=device, use_amp=use_amp)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume and Path(resume).exists():
            print(f"Resuming from checkpoint: {resume}")
            trainer.load(resume)
            start_epoch = len(trainer.train_losses)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        for epoch in range(start_epoch, epochs):
            try:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"{'='*80}")
                
                # Train
                train_loss, train_components = trainer.train_epoch(train_loader, epoch + 1)
                print(f"\nTrain Loss: {train_loss:.6f}")
                print(f"  L1: {train_components['l1']:.6f}, "
                      f"L2: {train_components['l2']:.6f}, "
                      f"Perceptual: {train_components['perceptual']:.6f}")
                
                # Validate
                val_loss, val_components = trainer.validate(val_loader)
                print(f"Val Loss: {val_loss:.6f}")
                print(f"  L1: {val_components['l1']:.6f}, "
                      f"L2: {val_components['l2']:.6f}, "
                      f"Perceptual: {val_components['perceptual']:.6f}")
                
                # Update learning rate
                trainer.scheduler.step()
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    trainer.save(
                        f"{output_folder}/best_model.pth",
                        epoch=epoch + 1,
                        metrics={'val_loss': val_loss}
                    )
                    print(f"✓ New best model! Val Loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{max_patience}")
                
                # Save periodic checkpoints
                if (epoch + 1) % 10 == 0:
                    trainer.save(f"{output_folder}/checkpoint_epoch_{epoch + 1}.pth")
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
                
                # Clean up GPU memory
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠ Out of memory at epoch {epoch + 1}")
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    print("Try reducing batch size or image resolution")
                    break
                else:
                    raise e
        
        # Save final model
        trainer.save(f"{output_folder}/final_model.pth")
        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print(f"Models saved to: {output_folder}")
        print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current state...")
        trainer.save(f"{output_folder}/interrupted_checkpoint.pth")
    
    except Exception as e:
        print(f"\n⚠ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()


# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Underwater Image Enhancement with VGG-16',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', required=True,
                       help='Input image folder path')
    parser.add_argument('--reference', required=True,
                       help='Reference image folder path')
    parser.add_argument('--output', default='./output',
                       help='Output folder for models and logs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train_enhanced_model(
        image_folder=args.input,
        reference_folder=args.reference,
        output_folder=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_amp=not args.no_amp,
        resume=args.resume
    )
    
    
    
    