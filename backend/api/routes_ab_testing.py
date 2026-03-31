from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.src.unit3_hypothesis_testing.ab_test_engine import ABTestEngine
from backend.src.unit3_hypothesis_testing.bayesian_estimation import BayesianABTest

router = APIRouter()


class FrequentistRequest(BaseModel):
    control_clicks: int = Field(..., ge=0, description="Clicks in control group")
    control_total: int = Field(..., gt=0, description="Total users in control group")
    variant_clicks: int = Field(..., ge=0, description="Clicks in variant group")
    variant_total: int = Field(..., gt=0, description="Total users in variant group")
    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level")


class SampleSizeRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0, lt=1, description="Current conversion rate")
    min_detectable_effect: float = Field(..., gt=0, description="Minimum effect to detect")
    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level")
    power: float = Field(0.8, gt=0, lt=1, description="Statistical power")


class BayesianRequest(BaseModel):
    control_successes: int = Field(..., ge=0)
    control_failures: int = Field(..., ge=0)
    variant_successes: int = Field(..., ge=0)
    variant_failures: int = Field(..., ge=0)
    prior_alpha: float = Field(1.0, gt=0)
    prior_beta: float = Field(1.0, gt=0)
    n_simulations: int = Field(100000, ge=1000, le=1000000)


@router.post("/test/frequentist")
def run_frequentist_test(request: FrequentistRequest):
    """Run a two-proportion z-test for A/B testing."""
    try:
        engine = ABTestEngine()
        result = engine.run_ztest(
            control_clicks=request.control_clicks,
            control_total=request.control_total,
            variant_clicks=request.variant_clicks,
            variant_total=request.variant_total,
            alpha=request.alpha,
        )
        result["confidence_interval"] = list(result["confidence_interval"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/test/sample-size")
def calculate_sample_size(request: SampleSizeRequest):
    """Calculate the required sample size per group for an A/B test."""
    try:
        engine = ABTestEngine()
        n = engine.calculate_sample_size(
            baseline_rate=request.baseline_rate,
            min_detectable_effect=request.min_detectable_effect,
            alpha=request.alpha,
            power=request.power,
        )
        return {"sample_size_per_group": n}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/test/bayesian")
def run_bayesian_test(request: BayesianRequest):
    """Run a Bayesian A/B test using Beta-Binomial conjugate model."""
    try:
        bayes = BayesianABTest(
            prior_alpha=request.prior_alpha,
            prior_beta=request.prior_beta,
        )
        bayes.update("A", request.control_successes, request.control_failures)
        bayes.update("B", request.variant_successes, request.variant_failures)

        prob_b_beats_a = bayes.probability_b_beats_a(
            n_simulations=request.n_simulations
        )
        loss_a = bayes.expected_loss("A", n_simulations=request.n_simulations)
        loss_b = bayes.expected_loss("B", n_simulations=request.n_simulations)
        posterior_a = bayes.get_posterior("A")
        posterior_b = bayes.get_posterior("B")

        return {
            "probability_b_beats_a": prob_b_beats_a,
            "expected_loss_a": loss_a,
            "expected_loss_b": loss_b,
            "posterior_a": {"alpha": posterior_a[0], "beta": posterior_a[1]},
            "posterior_b": {"alpha": posterior_b[0], "beta": posterior_b[1]},
            "recommendation": "Variant B" if prob_b_beats_a > 0.95 else (
                "Variant A" if prob_b_beats_a < 0.05 else "Collect more data"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
