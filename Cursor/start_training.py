#!/usr/bin/env python3
"""
Quick Start Training Script for FloorPlan Detection
اسکریپت شروع سریع تمرین برای تشخیص نقشه معماری

Usage:
    python start_training.py --model yolo --data path/to/data --epochs 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.floorplan_yolo import FloorplanYOLO
from data import DataManager
from config.logging_config import get_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FloorPlan Object Detection Training')
    
    parser.add_argument('--model', type=str, default='yolo', 
                       choices=['yolo', 'rtdetr'], help='Model architecture')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Config file path')
    parser.add_argument('--data', type=str, help='Data directory path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_model(config, device):
    """Create and initialize model"""
    logger = get_logger("model_creation")
    
    try:
        model = FloorplanYOLO(
            num_classes=config['data']['num_classes'],
            width_multiple=1.0,  # Can be adjusted for model size
            depth_multiple=1.0,
            input_size=tuple(config['data']['image_size'])
        )
        
        model = model.to(device)
        
        # Print model info
        info = model.get_model_info()
        logger.info(f"Model created successfully:")
        logger.info(f"  Parameters: {info['total_parameters']:,}")
        logger.info(f"  Model size: {info['model_size_mb']:.1f} MB")
        logger.info(f"  Input size: {info['input_size']}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise


def quick_data_test(data_manager):
    """Quick test of data pipeline"""
    logger = get_logger("data_test")
    
    try:
        # Setup datasets
        data_manager.setup_datasets()
        
        # Get data loaders
        dataloaders = data_manager.get_dataloaders()
        
        # Test one batch
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        logger.info(f"Data test successful:")
        logger.info(f"  Batch size: {batch['images'].shape[0]}")
        logger.info(f"  Image shape: {batch['images'].shape}")
        logger.info(f"  Number of targets: {len(batch['targets'])}")
        
        return dataloaders
        
    except Exception as e:
        logger.error(f"Data test failed: {e}")
        raise


def simple_training_loop(model, dataloaders, device, epochs=5):
    """Simple training loop for testing"""
    logger = get_logger("training")
    
    # Simple loss (for testing - you'll implement proper loss later)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = dataloaders['train']
    
    logger.info(f"Starting simple training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # Just test first 5 batches
                break
                
            try:
                images = batch['images'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Simple loss calculation (placeholder)
                # In real training, you'll use proper YOLO loss
                dummy_target = torch.zeros_like(outputs[0]['reg']).to(device)
                loss = criterion(outputs[0]['reg'], dummy_target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 2 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    logger.info("Simple training completed successfully!")


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logger = get_logger("main")
    logger.info("Starting FloorPlan Detection Training")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Override config with command line args
        if args.data:
            config['data']['dataset_path'] = args.data
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        
        # Setup device
        device = setup_device(args.device)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, device)
        
        # Setup data
        logger.info("Setting up data pipeline...")
        data_manager = DataManager(config)
        
        # Quick data test
        logger.info("Testing data pipeline...")
        dataloaders = quick_data_test(data_manager)
        
        # Simple training test
        if args.debug:
            logger.info("Running simple training test...")
            simple_training_loop(model, dataloaders, device, epochs=3)
        else:
            logger.info("Model and data setup completed successfully!")
            logger.info("Ready for full training implementation.")
            logger.info("Next steps:")
            logger.info("1. Implement proper YOLO loss function")
            logger.info("2. Add validation loop")
            logger.info("3. Add model checkpointing")
            logger.info("4. Add metrics calculation")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 