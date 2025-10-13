"""
ECLIPSE Framework - Simple Example
===================================
Demonstrates ECLIPSE with a toy classification problem
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from eclipse_core import EclipseFramework, EclipseConfig, FalsificationCriteria


def main():
    print("ECLIPSE SIMPLE EXAMPLE - Binary Classification")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Create data identifiers
    data_ids = list(range(len(X)))
    
    # =========================================================================
    # CONFIGURE ECLIPSE
    # =========================================================================
    
    config = EclipseConfig(
        project_name="SimpleClassification",
        researcher="Your Name",
        sacred_seed=2025,
        development_ratio=0.7,
        holdout_ratio=0.3,
        n_folds_cv=5,
        output_dir="./eclipse_simple_results"
    )
    
    eclipse = EclipseFramework(config)
    
    # =========================================================================
    # STAGE 1: SPLIT
    # =========================================================================
    
    dev_ids, holdout_ids = eclipse.stage1_irreversible_split(data_ids)
    
    dev_X = X[dev_ids]
    dev_y = y[dev_ids]
    holdout_X = X[holdout_ids]
    holdout_y = y[holdout_ids]
    
    # =========================================================================
    # STAGE 2: REGISTER CRITERIA
    # =========================================================================
    
    criteria = [
        FalsificationCriteria(
            name="f1_score",
            threshold=0.70,
            comparison=">=",
            description="F1-score must be at least 0.70",
            is_required=True
        ),
        FalsificationCriteria(
            name="precision",
            threshold=0.65,
            comparison=">=",
            description="Precision must be at least 0.65",
            is_required=True
        ),
        FalsificationCriteria(
            name="recall",
            threshold=0.65,
            comparison=">=",
            description="Recall must be at least 0.65",
            is_required=True
        )
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    # =========================================================================
    # STAGE 3: DEVELOPMENT
    # =========================================================================
    
    def train_function(train_data, **kwargs):
        """Train a random forest classifier"""
        train_indices = train_data
        X_train = dev_X[train_indices]
        y_train = dev_y[train_indices]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def validation_function(model, val_data, **kwargs):
        """Validate model and return metrics"""
        val_indices = val_data
        X_val = dev_X[val_indices]
        y_val = dev_y[val_indices]
        
        y_pred = model.predict(X_val)
        
        return {
            'f1_score': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred)
        }
    
    # Prepare development data (just indices)
    dev_data = list(range(len(dev_X)))
    
    dev_results = eclipse.stage3_development(
        development_data=dev_data,
        training_function=train_function,
        validation_function=validation_function
    )
    
    # =========================================================================
    # STAGE 4: SINGLE-SHOT VALIDATION
    # =========================================================================
    
    # Train final model on all development data
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(dev_X, dev_y)
    
    def holdout_validation_function(model, holdout_data, **kwargs):
        """Validate on holdout (NEVER SEEN BEFORE)"""
        y_pred = model.predict(holdout_data)
        y_true = holdout_y
        
        return {
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }
    
    validation_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_X,
        final_model=final_model,
        validation_function=holdout_validation_function
    )
    
    if validation_results is None:
        print("Validation was cancelled.")
        return
    
    # =========================================================================
    # STAGE 5: FINAL ASSESSMENT
    # =========================================================================
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=validation_results
    )
    
    # =========================================================================
    # VERIFY INTEGRITY
    # =========================================================================
    
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("ECLIPSE SIMPLE EXAMPLE COMPLETE")
    print(f"Final verdict: {final_assessment['verdict']}")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()