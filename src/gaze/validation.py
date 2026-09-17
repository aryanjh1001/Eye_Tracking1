import math
from dataclasses import dataclass
from typing import List
from .calibration import CollectedSample
from .mapper import GazeMapper

@dataclass
class ValidationMetrics:
    mean_error: float
    median_error: float
    max_error: float
    validation_targets_excluded_from_fitting: bool
    train_targets_count: int
    val_targets_count: int
    train_samples_count: int
    val_samples_count: int

def compute_validation_metrics(mapper: GazeMapper, 
                             val_samples: List[CollectedSample], 
                             train_samples: List[CollectedSample], 
                             screen_width: int, 
                             screen_height: int) -> ValidationMetrics:
                                 
    if not val_samples:
        return ValidationMetrics(
            0.0, 0.0, 0.0,
            validation_targets_excluded_from_fitting=True,
            train_targets_count=len(set((s.target_x, s.target_y) for s in train_samples)),
            val_targets_count=0,
            train_samples_count=len(train_samples),
            val_samples_count=0
        )
        
    errors = []
    for s in val_samples:
        pred_x, pred_y = mapper.predict(s.features, screen_width, screen_height)
        err = math.sqrt((pred_x - s.target_x)**2 + (pred_y - s.target_y)**2)
        errors.append(err)
        
    errors.sort()
    
    mean_err = sum(errors) / len(errors)
    median_err = errors[len(errors)//2] if len(errors) % 2 != 0 else (errors[len(errors)//2 - 1] + errors[len(errors)//2]) / 2.0
    max_err = errors[-1]
    
    train_targets = set((s.target_x, s.target_y) for s in train_samples)
    val_targets = set((s.target_x, s.target_y) for s in val_samples)
    
    # Verify disjoint
    is_disjoint = train_targets.isdisjoint(val_targets)
    
    return ValidationMetrics(
        mean_error=mean_err,
        median_error=median_err,
        max_error=max_err,
        validation_targets_excluded_from_fitting=is_disjoint,
        train_targets_count=len(train_targets),
        val_targets_count=len(val_targets),
        train_samples_count=len(train_samples),
        val_samples_count=len(val_samples)
    )
