import argparse
import json
import numpy as np

from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter,
)

from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TuningStep, TrainingStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--role")
    parser.add_argument("--image-uri")
    parser.add_argument("--pipeline-name")
    parser.add_argument("--source-s3-uri")
    parser.add_argument("--preprocessing-script-s3")
    parser.add_argument("--evaluation-script-s3")
    parser.add_argument("--definition-output")

    args, _ = parser.parse_known_args()

    sagemaker_session = PipelineSession()
    bucket = sagemaker_session.default_bucket()
    role = args.role
    pipeline_name = args.pipeline_name
    image_uri = args.image_uri

    dataset_param = ParameterString(
        name="DatasetName",
        enum_values=["breast-cancer", "banknote-authentication"],
    )
    stratify_train_test_split = ParameterString(
        name="StratifySplit",
        default_value="True",
        enum_values=["True", "False"],
    )
    preprocess_testset_size = ParameterFloat(
        name="TestSetSize", default_value=0.2)
    preprocess_random_state = ParameterInteger(
        name="RandomState", default_value=42)

    preprocessor = ScriptProcessor(
        image_uri=image_uri,
        command=["python"],
        instance_type="ml.t3.xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_args = preprocessor.run(
        arguments=[
            "--dataset", dataset_param,
            "--target", "CLASS",
            "--test-size", preprocess_testset_size.to_string(),
            "--stratify", stratify_train_test_split,
            "--random-state", preprocess_random_state.to_string(),
            "--file-name", "dataset.csv",
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
            ),
        ],
        code=args.preprocessing_script_s3,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        step_args=step_args
    )

    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    tuning_cross_validation_scorer = ParameterString(
        name="CrossValidationScorer",
        default_value="roc_auc",
        enum_values=[
            'accuracy',
            'balanced_accuracy',
            'top_k_accuracy',
            'average_precision',
            'f1',
            'f1_micro',
            'f1_macro',
            'f1_weighted',
            'precision',
            'recall',
            'jaccard',
            'roc_auc',
            'roc_auc_ovr',
            'roc_auc_ovo',
            'roc_auc_ovr_weighted',
            'roc_auc_ovo_weighted',
        ],
    )

    common_hyperparameters = {
        "target": "CLASS",
        "cross-validation": True,
        "train-file": "dataset.csv",
        "cv-scorer": tuning_cross_validation_scorer,
    }

    metric_definitions = [
        {"Name": "val_score", "Regex": "Cross-Validation Score: ([0-9.]+).*$"},
    ]

    estimator_dict = {
        "RandomForest": SKLearn(
            entry_point="estimators/RandomForest.py",
            source_dir=args.source_s3_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            image_uri=image_uri,
            # need to pass copy because each job updates this dictionary and all uses the same reference
            hyperparameters=common_hyperparameters.copy(),
            sagemaker_session=sagemaker_session,
        ),
        "NeuralNet": SKLearn(
            entry_point="estimators/NeuralNet.py",
            source_dir=args.source_s3_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            image_uri=image_uri,
            hyperparameters=common_hyperparameters.copy(),
            sagemaker_session=sagemaker_session,
        ),
        "LogisticRegression": SKLearn(
            entry_point="estimators/LogisticRegression.py",
            source_dir=args.source_s3_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            image_uri=image_uri,
            hyperparameters=common_hyperparameters.copy(),
            sagemaker_session=sagemaker_session,
        ),
        "NaiveBayes": SKLearn(
            entry_point="estimators/NaiveBayes.py",
            source_dir=args.source_s3_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            image_uri=image_uri,
            hyperparameters=common_hyperparameters.copy(),
            sagemaker_session=sagemaker_session,
        ),
        "XGBoost": SKLearn(
            entry_point="estimators/XGBoost.py",
            source_dir=args.source_s3_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            image_uri=image_uri,
            hyperparameters=common_hyperparameters.copy(),
            sagemaker_session=sagemaker_session,
        ),
        "LightGBM": SKLearn(
            entry_point="estimators/LightGBM.py",
            source_dir=args.source_s3_uri,
            role=role,
            instance_count=1,
            instance_type="ml.m5.large",
            image_uri=image_uri,
            hyperparameters=common_hyperparameters.copy(),
            sagemaker_session=sagemaker_session,
        )
    }

    hyperparameter_ranges_dict = {
        "RandomForest": {
            "n-estimators": CategoricalParameter(list(range(50, 501, 50))),
            "criterion": CategoricalParameter(["gini", "entropy"]),
            "max-depth": CategoricalParameter(list(range(5, 21, 5))),
            "min-samples-split": IntegerParameter(2, 11),
            "min-samples-leaf": IntegerParameter(1, 11),
            "max-features": CategoricalParameter(['sqrt', 'log2', None]),
            "class-weight": CategoricalParameter(["balanced", None]),
        },
        "NeuralNet": {
            "hidden-layer-sizes": CategoricalParameter([50, 100, "50,50"]),
            "activation": CategoricalParameter(['logistic', 'relu']),
            "learning-rate-init": CategoricalParameter(np.logspace(-3, -1, num=5).tolist()),
            "alpha": CategoricalParameter(np.logspace(-5, 3, num=5).tolist()),
            "max-iter": CategoricalParameter([100, 200, 500, 1000]),
        },
        "LogisticRegression": {
            "penalty": CategoricalParameter(["l2", None]),
            # Max 30 allowed in `CategoricalParameter`
            "C": CategoricalParameter(np.logspace(-3, 3, num=30).tolist()),
            # "solver": CategoricalParameter(["liblinear", "lbfgs"]),  # According to penalty lbfgs is set to default in training script
            "class-weight": CategoricalParameter(["balanced", None]),
            "max-iter": CategoricalParameter([50, 100, 200, 500]),
        },
        "NaiveBayes": {
            "var_smoothing": CategoricalParameter(np.logspace(-10, 0, num=30).tolist()),
        },
        "XGBoost": {
            "booster": CategoricalParameter(['gbtree', 'gblinear', 'dart']),
            "n-estimators": CategoricalParameter(list(range(50, 201, 50))),
            "learning-rate": CategoricalParameter(np.logspace(-3, -1, num=5).tolist()),
            "max-depth": IntegerParameter(3, 7),
            "min-child-weight": IntegerParameter(1, 10),
            "subsample": CategoricalParameter(np.linspace(0.6, 1, num=5).tolist()),
            "colsample-bytree": CategoricalParameter(np.linspace(0.5, 1, num=5).tolist()),
            "gamma": CategoricalParameter([0, 0.1, 0.5]),
            "reg-alpha": CategoricalParameter([0, 0.01, 0.1, 1]),
            "reg-lambda": CategoricalParameter([0.01, 0.1, 1]),
        },
        "LightGBM": {
            "boosting-type": CategoricalParameter(['gbdt', 'dart', 'goss']),
            "num-leaves": IntegerParameter(10, 101),
            "max-depth": IntegerParameter(3, 7),
            "learning-rate": CategoricalParameter(np.logspace(-3, -1, num=5).tolist()),
            "n-estimators": CategoricalParameter(list(range(50, 201, 50))),
            "subsample": CategoricalParameter(np.linspace(0.5, 1, num=5).tolist()),
            "colsample-bytree": CategoricalParameter(np.linspace(0.5, 1, num=5).tolist()),
            "reg-alpha": CategoricalParameter([0, 0.1, 0.5]),
            "reg-lambda": CategoricalParameter([0, 0.1, 0.5]),
            "min-split-gain": CategoricalParameter([0, 0.1, 0.5]),
            "min-child-weight": IntegerParameter(1, 10),
            "min-child-samples": IntegerParameter(5, 21),
        },
    }

    objective_metric_name_dict = {k: "val_score" for k in estimator_dict}
    metric_definitions_dict = {k: metric_definitions for k in estimator_dict}

    tuning_max_jobs = ParameterInteger(
        name="MaxTuningJobs", default_value=10*len(estimator_dict))
    tuning_max_parallel_jobs = ParameterInteger(
        name="MaxTuningParallelJobs", default_value=len(estimator_dict))

    optimizer = HyperparameterTuner.create(
        estimator_dict=estimator_dict,
        hyperparameter_ranges_dict=hyperparameter_ranges_dict,
        objective_metric_name_dict=objective_metric_name_dict,
        metric_definitions_dict=metric_definitions_dict,
        objective_type="Maximize",
        max_jobs=tuning_max_jobs,
        max_parallel_jobs=tuning_max_parallel_jobs,
        strategy="Random",
    )

    train_inputs = {
        "train": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
        ),
    }

    step_args = optimizer.fit(
        inputs={k: train_inputs for k in estimator_dict},
        include_cls_metadata={k: False for k in estimator_dict}
    )

    step_tuning = TuningStep(
        name="TuningWithCrossValidation",
        step_args=step_args,
        depends_on=[step_preprocess,],
    )

    best_estimator_param = {
        "best-training-job": step_tuning.properties.BestTrainingJob.TrainingJobName,
        "best-algorithm": step_tuning.properties.BestTrainingJob.TrainingJobDefinitionName,
        "aws-region": sagemaker_session.boto_region_name
    }

    best_estimator = SKLearn(
        entry_point="refit.py",
        source_dir=args.source_s3_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        image_uri=image_uri,
        hyperparameters=best_estimator_param,
        metric_definitions=metric_definitions,
        sagemaker_session=sagemaker_session,
    )

    step_args = best_estimator.fit(train_inputs)

    step_refit = TrainingStep(
        name="RefitBestModel",
        step_args=step_args,
        depends_on=[step_tuning,],
    )

    register_eval_metric = ParameterString(
        name="MetricForRegistrationThreshold",
        default_value="auc",
        enum_values=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
        ],
    )
    register_eval_metric_threshold = ParameterFloat(
        name="MinThesholdForRegisterMetric",
        default_value=0.7,
    )

    register_model_approval_status = ParameterString(
        name="RegisterModelApprovalStatus",
        default_value="PendingManualApproval",
        enum_values=[
            "PendingManualApproval",
            "Approved",
        ],
    )

    evaluate_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python"],
        instance_type="ml.t3.xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_args = evaluate_processor.run(
        arguments=[
            "--target", "CLASS",
            "--test-file", "dataset.csv",
            "--register-metric", register_eval_metric,
        ],
        inputs=[
            ProcessingInput(
                source=step_refit.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
        code=args.evaluation_script_s3,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    step_evaluate = ProcessingStep(
        name="EvaluateBestModel",
        step_args=step_args,
        property_files=[evaluation_report,],
        depends_on=[step_refit,],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_evaluate.properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    model = SKLearnModel(
        model_data=step_refit.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point='inference.py',
        source_dir=args.source_s3_uri,
        image_uri=best_estimator.training_image_uri(),
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=Join(
            on="-", values=[pipeline_name, dataset_param]),
        model_metrics=model_metrics,
        approval_status=register_model_approval_status,
    )

    step_register = ModelStep(
        name="RegisterBestModel",
        step_args=step_args,
        depends_on=[step_evaluate,],
    )

    step_fail = FailStep(
        name="RegisterFailed",
        error_message=Join(
            on=" ",
            values=[
                "Execution failed because",
                register_eval_metric,
                "value is less than the required",
                register_eval_metric_threshold,
            ]
        ),
    )

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="register_metric.value",
            # json_path="binary_classification_metrics.accuracy.value",
        ),
        right=register_eval_metric_threshold,
    )

    step_cond = ConditionStep(
        name="RegisterConditionCheck",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            dataset_param,
            stratify_train_test_split,
            preprocess_testset_size,
            preprocess_random_state,
            tuning_cross_validation_scorer,
            tuning_max_jobs,
            tuning_max_parallel_jobs,
            register_eval_metric,
            register_eval_metric_threshold,
            register_model_approval_status,
        ],
        steps=[
            step_preprocess, step_tuning, step_refit,
            step_evaluate, step_cond,
        ],
        sagemaker_session=sagemaker_session,
    )

    pipeline_definition = json.loads(pipeline.definition())
    with open(args.definition_output, "w") as fw:
        json.dump(pipeline_definition, fw)
