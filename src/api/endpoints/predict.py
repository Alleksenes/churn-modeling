# ============================================
# File: src/api/endpoints/predict.py
# ============================================
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, status
from loguru import logger

from src.api.schemas import (
    BulkChurnPredictionInput,
    BulkChurnPredictionOutput,
    ChurnPredictionInput,
    ChurnPredictionOutput,
)
from src.churn_model.predict import make_prediction

router = APIRouter()


@router.post(
    "/predict",
    response_model=ChurnPredictionOutput,
    summary="Predict Churn for a Single Customer",
    tags=["Prediction"],
    description="Accepts data for a single customer and returns the churn prediction and probability.",
)
async def post_predict_single(
    input_data: ChurnPredictionInput = Body(...),
) -> ChurnPredictionOutput:
    logger.info("Received single prediction request.")

    try:
        input_df = pd.DataFrame([input_data.model_dump()])
        results = make_prediction(input_data=input_df)

        if error_msg := results.get("error"):
            logger.error(f"Prediction function returned error: {error_msg}")
            if (
                "Model not loaded" in error_msg
                or "unavailable" in error_msg
                or "initialization failed" in error_msg
            ):
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif "Invalid input" in error_msg or "Missing input" in error_msg:
                status_code = status.HTTP_400_BAD_REQUEST
            else:  # General prediction failure
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise HTTPException(status_code=status_code, detail=error_msg)

        prediction = results.get("predictions", [None])[0]
        probability = results.get("probabilities", [None])[0]

        if prediction is None:
            logger.error("Prediction successful but result prediction is None.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction resulted in null value.",
            )

        logger.info(
            f"Single prediction successful: Prediction={prediction}, Probability={probability}"
        )
        return ChurnPredictionOutput(
            prediction=prediction, probability_churn=probability, error=None
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error during single prediction endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
        ) from e


@router.post(
    "/predict/bulk",
    response_model=BulkChurnPredictionOutput,
    summary="Predict Churn for Multiple Customers",
    tags=["Prediction"],
    description="Accepts a list of customer data objects & returns predictions.",
)
async def post_predict_bulk(
    input_data: BulkChurnPredictionInput = Body(...),
) -> BulkChurnPredictionOutput:
    logger.info(
        f"Received bulk prediction request for {len(input_data.inputs)} inputs."
    )

    try:
        input_df = pd.DataFrame([item.model_dump() for item in input_data.inputs])
        results = make_prediction(input_data=input_df)
        if error_msg := results.get("error"):
            logger.error(f"Bulk prediction function returned error: {error_msg}")
            if (
                "Model not loaded" in error_msg
                or "unavailable" in error_msg
                or "initialization failed" in error_msg
            ):
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif "Invalid input" in error_msg or "Missing input" in error_msg:
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise HTTPException(
                status_code=status_code, detail=f"Bulk prediction failed: {error_msg}"
            )

        output_results = []
        predictions = results.get("predictions", [])
        probabilities = results.get("probabilities")

        if len(predictions) != len(input_data.inputs):
            logger.error(
                f"Mismatch between input count ({len(input_data.inputs)}) and prediction count ({len(predictions)})."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: Prediction count mismatch.",
            )

        for i, pred in enumerate(predictions):
            prob = (
                probabilities[i] if probabilities and i < len(probabilities) else None
            )
            output_results.append(
                ChurnPredictionOutput(
                    prediction=pred, probability_churn=prob, error=None
                )
            )

        logger.info(f"Bulk prediction successful for {len(output_results)} inputs.")
        return BulkChurnPredictionOutput(results=output_results)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error during bulk prediction endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred during bulk prediction.",
        ) from e
