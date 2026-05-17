"""
Detection Service - Core ML inference interface.

Manages session-centric Transformer model for behavioral detection.
"""

import torch
import numpy as np
import joblib
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path

from app.core.config import settings
from src.data_loader.feature_transforms import FeatureTransformConfig, apply_feature_transforms
from src.data_loader.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


class DetectionService:
    """Service for running CyberShield behavioral detection."""
    
    def __init__(self):
        """Initialize detection service with checkpoint and scaler."""
        self.device = torch.device(settings.DEVICE)
        self.checkpoint_path = Path(settings.CHECKPOINT_PATH)
        self.scaler_path = Path(settings.SCALER_PATH)
        
        self.model = None
        self.scaler = None
        self.feature_transform_config = FeatureTransformConfig()
        self.feature_normalizer = None
        self.session_len = 20
        self.feature_dim = 10
        self._is_initialized = False
        
    def initialize(self):
        """Load model and scaler from disk."""
        if self._is_initialized:
            return
            
        try:
            # Load checkpoint
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
                
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            
            # Import model class
            from src.models.transformer import C2Transformer
            
            # Extract model configuration from checkpoint metadata
            use_derivative_features = checkpoint.get('use_derivative_features', False)
            self.feature_dim = int(checkpoint.get('feature_dim', 10))
            self.session_len = int(checkpoint.get('session_len', 20))
            self.feature_transform_config = FeatureTransformConfig.from_checkpoint_dict(
                checkpoint.get('feature_transform_config')
            )
            if checkpoint.get('normalize_features', False) and checkpoint.get('feature_normalizer'):
                self.feature_normalizer = FeatureNormalizer.from_checkpoint_dict(checkpoint['feature_normalizer'])
            
            # Initialize model with checkpoint configuration
            self.model = C2Transformer(
                feature_dim=self.feature_dim,
                seq_len=self.session_len,
                d_model=64,          # Hidden dimension
                nhead=4,             # Number of attention heads
                num_layers=3,        # Transformer encoder layers
                dim_feedforward=128, # FFN dimension
                dropout=0.2,         # Dropout rate
                use_derivative_features=use_derivative_features
            )
            
            # Load model weights - checkpoint should have 'model_state_dict' key
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded scaler from {self.scaler_path}")
            else:
                logger.warning(f"Scaler not found: {self.scaler_path}")
                
            self._is_initialized = True
            logger.info("Detection service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize detection service: {e}")
            raise
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Extract 10 features from session data."""
        try:
            # Feature extraction (matching src/features/feature_config.py schema)
            features = np.array([
                session_data.get("duration", 0.0),
                session_data.get("bytes_in", 0.0),
                session_data.get("bytes_out", 0.0),
                session_data.get("packets_in", 0),
                session_data.get("packets_out", 0),
                1.0 if session_data.get("protocol", "").upper() == "TCP" else 0.0,
                session_data.get("src_port", 0) / 65535.0,  # Normalize
                session_data.get("dst_port", 0) / 65535.0,  # Normalize
                session_data.get("timestamp", 0.0),
                0.0,  # Reserved feature
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def infer_batch(self, sessions: List[Dict[str, Any]]) -> List[Tuple[float, bool, float]]:
        """
        Run inference on a batch of sessions.
        
        Returns:
            List of (risk_score, is_suspicious, confidence) tuples
        """
        if not self._is_initialized:
            self.initialize()
            
        try:
            features = np.array([self.extract_features(session) for session in sessions], dtype=np.float32)
            sequences = np.zeros((len(sessions), self.session_len, self.feature_dim), dtype=np.float32)
            sequences[:, 0, :] = features[:, : self.feature_dim]
            real_masks = np.zeros((len(sessions), self.session_len), dtype=bool)
            real_masks[:, 0] = True
            return self._predict_from_sequences(sequences, real_masks)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def infer_tensor_batch(self, sequences: np.ndarray, real_masks: np.ndarray) -> List[Tuple[float, bool, float]]:
        """Run inference on a batch of session tensors with real-flow masks."""
        if not self._is_initialized:
            self.initialize()

        try:
            return self._predict_from_sequences(sequences, real_masks)
        except Exception as e:
            logger.error(f"Tensor inference failed: {e}")
            raise

    def _predict_from_sequences(
        self,
        sequences: np.ndarray,
        real_masks: np.ndarray,
    ) -> List[Tuple[float, bool, float]]:
        """Apply checkpoint preprocessing and run the model."""
        if sequences.ndim != 3:
            raise ValueError(f"Expected sequences with shape (B, L, D), got {sequences.shape}")

        processed = sequences.astype(np.float32, copy=True)
        valid_masks = real_masks.astype(bool)
        
        # Pad to model's expected feature dimension if necessary
        if processed.shape[-1] < self.feature_dim:
            logger.info(f"Padding features from {processed.shape[-1]} to {self.feature_dim}")
            pad_width = ((0, 0), (0, 0), (0, self.feature_dim - processed.shape[-1]))
            processed = np.pad(processed, pad_width, mode='constant', constant_values=0.0)
            # Extend masks to match padded features
            mask_pad_width = ((0, 0), (0, 0), (0, self.feature_dim - real_masks.shape[-1]))
            valid_masks = np.pad(valid_masks, mask_pad_width, mode='constant', constant_values=0)
        elif processed.shape[-1] > self.feature_dim:
            logger.warning(f"Data has more features ({processed.shape[-1]}) than model expects ({self.feature_dim}), truncating")
            processed = processed[:, :, :self.feature_dim]

        # Only apply feature transforms if feature count matches
        if processed.shape[-1] == self.feature_dim:
            try:
                processed = apply_feature_transforms(processed, valid_masks, self.feature_transform_config)
            except Exception as e:
                logger.warning(f"Feature transforms failed: {e}, skipping transforms")
        
        # Apply normalization if available
        if self.feature_normalizer is not None:
            try:
                processed = self.feature_normalizer.transform(processed, valid_masks)
            except Exception as e:
                logger.warning(f"Feature normalization failed: {e}, skipping normalization")
        elif self.scaler is not None:
            try:
                flat = processed.reshape(-1, processed.shape[-1])
                valid_rows = valid_masks.reshape(-1)
                if valid_rows.any():
                    flat[valid_rows] = self.scaler.transform(flat[valid_rows])
                processed = flat.reshape(processed.shape)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}, skipping scaling")

        padding_mask = ~valid_masks
        features_tensor = torch.from_numpy(processed).to(self.device)
        padding_tensor = torch.from_numpy(padding_mask).to(self.device)

        with torch.no_grad():
            logits = self.model(features_tensor, padding_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        results = []
        for prob in probabilities:
            risk_score = float(prob)
            is_suspicious = risk_score > 0.5
            confidence = max(risk_score, 1 - risk_score)
            results.append((risk_score, is_suspicious, confidence))

        return results
    
    def infer_single(self, session: Dict[str, Any]) -> Tuple[float, bool, float]:
        """Run inference on a single session."""
        results = self.infer_batch([session])
        return results[0]
