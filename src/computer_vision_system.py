import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json
import base64
import io
from PIL import Image
import random
import os
from datetime import timedelta
try:
    from .db import upsert_heartbeat
except Exception:
    upsert_heartbeat = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DefectDetection:
    """Computer vision defect detection result"""
    detection_id: str
    train_id: str
    camera_id: str
    defect_type: str  # rust, crack, wear, damage, debris
    confidence: float
    bounding_box: Dict[str, int]  # x, y, width, height
    severity: str  # low, medium, high, critical
    timestamp: datetime
    image_path: str
    recommendations: List[str]

@dataclass
class InspectionResult:
    """Complete inspection result for a train"""
    inspection_id: str
    train_id: str
    timestamp: datetime
    total_defects: int
    critical_defects: int
    overall_condition: str  # excellent, good, fair, poor, critical
    confidence_score: float
    defect_detections: List[DefectDetection]
    maintenance_required: bool
    estimated_repair_time: int  # hours

class TrainDefectDetectionModel(nn.Module):
    """CNN model for train defect detection"""
    
    def __init__(self, num_classes: int = 6):  # normal, rust, crack, wear, damage, debris
        super(TrainDefectDetectionModel, self).__init__()
        
        # Use pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace final layer for our specific defect classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Additional layers for defect localization
        self.localization_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # x, y, width, height
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(self.backbone.layer2(
                self.backbone.layer1(self.backbone.maxpool(
                    self.backbone.relu(self.backbone.bn1(
                        self.backbone.conv1(x)
                    ))
                ))
            ))
        ))
        features = torch.flatten(features, 1)
        
        # Classification
        classification = self.backbone.fc(features)
        
        # Localization (bounding box)
        localization = self.localization_head(features)
        
        # Confidence score
        confidence = self.confidence_head(features)
        
        return classification, localization, confidence

class ComputerVisionSystem:
    """Computer vision system for automated train inspection"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Initialize model
        self.model = TrainDefectDetectionModel()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"✅ Loaded pre-trained model from {model_path}")
        else:
            logger.info("⚠️ No pre-trained model found. Using randomly initialized model for demo.")
        
        self.model.to(self.device)
        self.model.eval()
        try:
            if upsert_heartbeat:
                upsert_heartbeat('cv', 'ok', 'model_loaded')
        except Exception:
            pass
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Defect classes
        self.defect_classes = {
            0: 'normal',
            1: 'rust',
            2: 'crack', 
            3: 'wear',
            4: 'damage',
            5: 'debris'
        }
        
        # Severity mapping
        self.severity_mapping = {
            'normal': 'none',
            'rust': 'low',
            'crack': 'high',
            'wear': 'medium',
            'damage': 'critical',
            'debris': 'medium'
        }
        
        # Camera configurations for different train areas
        self.camera_configs = {
            'exterior_front': {'position': 'front', 'angle': 0, 'focus': 'body'},
            'exterior_side_left': {'position': 'left', 'angle': 90, 'focus': 'wheels'},
            'exterior_side_right': {'position': 'right', 'angle': 270, 'focus': 'wheels'},
            'exterior_rear': {'position': 'rear', 'angle': 180, 'focus': 'body'},
            'undercarriage': {'position': 'bottom', 'angle': -90, 'focus': 'mechanical'},
            'pantograph': {'position': 'top', 'angle': 90, 'focus': 'electrical'}
        }
        
        # Inspection history
        self.inspection_results = []
        
    def simulate_camera_image(self, train_id: str, camera_id: str) -> np.ndarray:
        """Simulate camera image capture for demo purposes"""
        # Create a synthetic train image with potential defects
        img_height, img_width = 480, 640
        
        # Create base train image (simplified representation)
        image = np.random.randint(50, 150, (img_height, img_width, 3), dtype=np.uint8)
        
        # Add train-like structure
        if 'exterior' in camera_id:
            # Add train body rectangle
            cv2.rectangle(image, (50, 150), (590, 350), (120, 120, 120), -1)
            # Add windows
            for i in range(4):
                x = 100 + i * 120
                cv2.rectangle(image, (x, 180), (x + 80, 250), (200, 200, 255), -1)
            
        elif 'undercarriage' in camera_id:
            # Add wheels and mechanical components
            cv2.circle(image, (150, 300), 50, (80, 80, 80), -1)
            cv2.circle(image, (490, 300), 50, (80, 80, 80), -1)
            
        # Randomly add defects for demo
        if random.random() < 0.3:  # 30% chance of defect
            defect_type = random.choice(['rust', 'crack', 'wear', 'damage', 'debris'])
            
            if defect_type == 'rust':
                # Add rust patches
                cv2.circle(image, (random.randint(100, 500), random.randint(200, 400)), 
                          random.randint(10, 30), (0, 50, 120), -1)
            elif defect_type == 'crack':
                # Add crack lines
                pt1 = (random.randint(100, 300), random.randint(200, 300))
                pt2 = (pt1[0] + random.randint(20, 100), pt1[1] + random.randint(-20, 20))
                cv2.line(image, pt1, pt2, (0, 0, 0), 2)
            elif defect_type == 'debris':
                # Add debris spots
                cv2.circle(image, (random.randint(100, 500), random.randint(300, 400)), 
                          random.randint(5, 15), (50, 30, 20), -1)
        
        return image
    
    def detect_defects(self, image: np.ndarray, train_id: str, camera_id: str) -> List[DefectDetection]:
        """Detect defects in train image using computer vision"""
        detections = []
        
        try:
            # Convert image to PIL and preprocess
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Run inference
                classification, localization, confidence = self.model(input_tensor)
                
                # Get predictions
                probs = F.softmax(classification, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                class_confidence = probs[0][predicted_class].item()
                overall_confidence = confidence[0].item()
                
                # Get bounding box coordinates
                bbox = localization[0].cpu().numpy()
                bbox_dict = {
                    'x': int(bbox[0] * image.shape[1]),
                    'y': int(bbox[1] * image.shape[0]),
                    'width': int(bbox[2] * image.shape[1]),
                    'height': int(bbox[3] * image.shape[0])
                }
                
                # Only create detection if defect is found (not normal)
                if predicted_class > 0 and class_confidence > 0.5:
                    defect_type = self.defect_classes[predicted_class]
                    severity = self.severity_mapping[defect_type]
                    
                    # Generate recommendations based on defect type
                    recommendations = self._generate_recommendations(defect_type, severity)
                    
                    detection = DefectDetection(
                        detection_id=f"det_{int(time.time())}_{random.randint(1000, 9999)}",
                        train_id=train_id,
                        camera_id=camera_id,
                        defect_type=defect_type,
                        confidence=class_confidence,
                        bounding_box=bbox_dict,
                        severity=severity,
                        timestamp=datetime.now(),
                        image_path=f"images/{train_id}_{camera_id}_{int(time.time())}.jpg",
                        recommendations=recommendations
                    )
                    
                    detections.append(detection)
                    
                    logger.info(f"🔍 Defect detected: {defect_type} on {train_id} (confidence: {class_confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Error in defect detection: {e}")
        
        return detections
    
    def _generate_recommendations(self, defect_type: str, severity: str) -> List[str]:
        """Generate maintenance recommendations based on defect type and severity"""
        recommendations = []
        
        base_recommendations = {
            'rust': [
                "Clean affected area and apply anti-rust treatment",
                "Inspect surrounding areas for corrosion spread",
                "Schedule preventive maintenance for similar components"
            ],
            'crack': [
                "Immediate structural inspection required",
                "Determine crack propagation risk",
                "Consider component replacement if critical",
                "Implement stress monitoring"
            ],
            'wear': [
                "Monitor wear progression over next inspections",
                "Plan component replacement within maintenance window",
                "Check lubrication systems"
            ],
            'damage': [
                "Detailed damage assessment required",
                "Safety evaluation before continued operation",
                "Emergency repair may be necessary"
            ],
            'debris': [
                "Remove debris from affected area",
                "Clean and inspect underlying surface",
                "Check for access control improvements"
            ]
        }
        
        recommendations.extend(base_recommendations.get(defect_type, []))
        
        # Add severity-specific recommendations
        if severity in ['high', 'critical']:
            recommendations.insert(0, "URGENT: Immediate maintenance action required")
            if severity == 'critical':
                recommendations.insert(1, "Consider taking train out of service")
        
        return recommendations
    
    def perform_full_inspection(self, train_id: str) -> InspectionResult:
        """Perform complete computer vision inspection of a train"""
        logger.info(f"🔍 Starting full inspection of train {train_id}")
        
        all_detections = []
        
        # Inspect each camera position
        for camera_id in self.camera_configs.keys():
            # Simulate image capture
            image = self.simulate_camera_image(train_id, camera_id)
            
            # Detect defects in image
            detections = self.detect_defects(image, train_id, camera_id)
            all_detections.extend(detections)
            
            # Small delay to simulate real camera processing
            time.sleep(0.5)
        
        # Analyze overall condition
        critical_defects = len([d for d in all_detections if d.severity == 'critical'])
        high_defects = len([d for d in all_detections if d.severity == 'high'])
        total_defects = len(all_detections)
        
        # Determine overall condition
        if critical_defects > 0:
            overall_condition = 'critical'
            maintenance_required = True
            estimated_repair_time = 24  # hours
        elif high_defects > 2:
            overall_condition = 'poor'
            maintenance_required = True
            estimated_repair_time = 12
        elif total_defects > 5:
            overall_condition = 'fair'
            maintenance_required = True
            estimated_repair_time = 6
        elif total_defects > 0:
            overall_condition = 'good'
            maintenance_required = False
            estimated_repair_time = 2
        else:
            overall_condition = 'excellent'
            maintenance_required = False
            estimated_repair_time = 0
        
        # Calculate confidence score
        if all_detections:
            confidence_score = sum([d.confidence for d in all_detections]) / len(all_detections)
        else:
            confidence_score = 0.95  # High confidence for no defects detected
        
        inspection_result = InspectionResult(
            inspection_id=f"insp_{train_id}_{int(time.time())}",
            train_id=train_id,
            timestamp=datetime.now(),
            total_defects=total_defects,
            critical_defects=critical_defects,
            overall_condition=overall_condition,
            confidence_score=confidence_score,
            defect_detections=all_detections,
            maintenance_required=maintenance_required,
            estimated_repair_time=estimated_repair_time
        )
        
        # Store inspection result
        self.inspection_results.append(inspection_result)
        
        logger.info(f"✅ Inspection complete for {train_id}: {overall_condition} condition, {total_defects} defects found")
        
        return inspection_result

    # Backward-compatible alias for tests expecting inspect_train()
    def inspect_train(self, train_id: str):
        result = self.perform_full_inspection(train_id)
        # Provide a dict view expected by some tests
        return {
            'train_id': result.train_id,
            'inspection_time': result.timestamp.isoformat(),
            'views_inspected': len(self.camera_configs),
            'defects_detected': result.total_defects,
            'overall_condition': result.overall_condition
        }
    
    def get_inspection_history(self, train_id: str = None, days: int = 30) -> List[InspectionResult]:
        """Get inspection history for analysis"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        results = [
            r for r in self.inspection_results 
            if r.timestamp >= cutoff_date
        ]
        
        if train_id:
            results = [r for r in results if r.train_id == train_id]
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def generate_inspection_report(self, inspection_result: InspectionResult) -> Dict[str, Any]:
        """Generate detailed inspection report"""
        report = {
            'inspection_summary': {
                'inspection_id': inspection_result.inspection_id,
                'train_id': inspection_result.train_id,
                'timestamp': inspection_result.timestamp.isoformat(),
                'overall_condition': inspection_result.overall_condition,
                'confidence_score': inspection_result.confidence_score,
                'maintenance_required': inspection_result.maintenance_required,
                'estimated_repair_time_hours': inspection_result.estimated_repair_time
            },
            'defect_summary': {
                'total_defects': inspection_result.total_defects,
                'critical_defects': inspection_result.critical_defects,
                'high_severity': len([d for d in inspection_result.defect_detections if d.severity == 'high']),
                'medium_severity': len([d for d in inspection_result.defect_detections if d.severity == 'medium']),
                'low_severity': len([d for d in inspection_result.defect_detections if d.severity == 'low'])
            },
            'defect_details': [
                {
                    'detection_id': d.detection_id,
                    'camera_id': d.camera_id,
                    'defect_type': d.defect_type,
                    'severity': d.severity,
                    'confidence': d.confidence,
                    'bounding_box': d.bounding_box,
                    'recommendations': d.recommendations
                }
                for d in inspection_result.defect_detections
            ],
            'maintenance_recommendations': self._generate_overall_recommendations(inspection_result),
            'report_generated': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_overall_recommendations(self, inspection_result: InspectionResult) -> List[str]:
        """Generate overall maintenance recommendations"""
        recommendations = []
        
        if inspection_result.critical_defects > 0:
            recommendations.append("CRITICAL: Immediate maintenance required before next operation")
            recommendations.append("Detailed safety inspection by certified technician")
            
        if inspection_result.overall_condition == 'poor':
            recommendations.append("Schedule comprehensive maintenance within 24 hours")
            
        elif inspection_result.overall_condition == 'fair':
            recommendations.append("Plan maintenance during next scheduled window")
            
        if inspection_result.total_defects > 10:
            recommendations.append("Consider extended maintenance program")
            recommendations.append("Review maintenance procedures and intervals")
            
        # Add specific recommendations based on defect patterns
        defect_types = [d.defect_type for d in inspection_result.defect_detections]
        
        if defect_types.count('rust') >= 3:
            recommendations.append("Implement enhanced corrosion protection measures")
            
        if defect_types.count('crack') >= 2:
            recommendations.append("Structural integrity assessment recommended")
            
        if defect_types.count('wear') >= 4:
            recommendations.append("Review component replacement schedule")
        
        return recommendations
