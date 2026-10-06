import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

math_500_n8 = LightevalTaskConfig(
    name="math_500_n8",
    suite=["custom"],
    prompt_function=prompt.math_500,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 8}),
    ],
    version=2,
)

TASKS_TABLE = [math_500_n8]
