SCORE_SYSTEM = {
    "math500": """You are an expert evaluator of model-generated math answers.
Your task is to estimate whether a human evaluator would need to modify the model answer.
Return only valid JSON. Do not include explanations, markdown, or extra text.""",
    "popqa": """You are an expert evaluator of model-generated PopQA answers.
Your task is to estimate whether a human evaluator would need to correct the model's short factual answer.
Return only valid JSON. Do not include explanations, markdown, or extra text.""",
    "mmlupro": """You are an expert evaluator of model-generated MMLU-Pro answers.
Your task is to estimate whether a human evaluator would need to correct the model's multiple-choice answer.
Return only valid JSON. Do not include explanations, markdown, or extra text.""",
    "gpqa": """You are an expert evaluator of model-generated GPQA answers.
Your task is to estimate whether a human evaluator would need to correct the model's graduate-level science multiple-choice answer.
Return only valid JSON. Do not include explanations, markdown, or extra text.""",
}

SCORE = {
    "math500": """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer likely contains a mathematical error, unsupported conclusion, missing final answer, ambiguity, or format problem.
- Low score means the answer is likely correct and would be confirmed without modification.

Rules:
- Use only the problem and the model answer below.
- Do not assume access to the gold answer.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Problem:
{question}

Model answer:
{model_answer}
""",
    "popqa": """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer is likely factually wrong, missing, too vague, over-specific, ambiguous, malformed, or unlikely to exactly match a valid PopQA answer alias.
- Low score means the answer is likely a correct short factual answer and would be confirmed without modification.

Rules:
- Use only the PopQA question and the model answer below.
- Do not assume access to the gold answer.
- The expected response is a concise answer such as an entity, title, place, date, number, or short phrase.
- Treat explanations, multiple conflicting answers, unsupported hedging, and non-answer text as evidence that human correction may be needed.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Question:
{question}

Model answer:
{model_answer}
""",
    "mmlupro": """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer is likely the wrong option, not a valid option letter, ambiguous, unsupported, or inconsistent with the question and choices.
- Low score means the answer is likely the correct single option and would be confirmed without modification.

Rules:
- Use only the MMLU-Pro question, answer options, and model answer below.
- Do not assume access to the gold answer.
- The expected response is one option letter such as A, B, C, ..., or J.
- Treat missing final answers, multiple conflicting choices, answer text without a clear option letter, and non-answer text as evidence that human correction may be needed.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Question and options:
{question}

Model answer:
{model_answer}
""",
    "gpqa": """Estimate the posterior edit probability for this prompt-response pair.

Definition:
- score = probability in [0, 1] that a human evaluator would modify the model answer.
- High score means the answer is likely the wrong option, not a valid option letter, ambiguous, unsupported, or inconsistent with the GPQA question and choices.
- Low score means the answer is likely the correct single option and would be confirmed without modification.

Rules:
- Use only the GPQA question, answer options, and model answer below.
- Do not assume access to the gold answer.
- The expected response is one option letter: A, B, C, or D.
- Treat missing final answers, multiple conflicting choices, answer text without a clear option letter, and non-answer text as evidence that human correction may be needed.
- Return only valid JSON in exactly this format:
  {{"score": 0.73}}

Question and options:
{question}

Model answer:
{model_answer}
""",
}

SELECT_SYSTEM = {
    "math500": """You are an expert review-selection system.
Your only task is to choose which model-generated answers should be sent for human correction.
Return only valid JSON. Do not solve the problems. Do not include explanations.""",
    "popqa": """You are an expert review-selection system for PopQA-style factual question answering.
Your only task is to choose which model-generated short answers should be sent for human correction.
Return only valid JSON. Do not answer the questions. Do not include explanations.""",
    "mmlupro": """You are an expert review-selection system for MMLU-Pro multiple-choice reasoning.
Your only task is to choose which model-generated option-letter answers should be sent for human correction.
Return only valid JSON. Do not solve the questions. Do not include explanations.""",
    "gpqa": """You are an expert review-selection system for GPQA graduate-level science multiple-choice reasoning.
Your only task is to choose which model-generated option-letter answers should be sent for human correction.
Return only valid JSON. Do not solve the questions. Do not include explanations.""",
}

SELECT = {
    "math500": """You are selecting examples for human review.

You will receive a batch of model-generated math answers. Select the items most likely to contain an error and therefore most worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the problem and the model answer.
- Do not use or assume the gold answer.
- Prefer examples with suspicious reasoning, arithmetic mistakes, missing final answers, format violations, or unsupported conclusions.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
""",
    "popqa": """You are selecting examples for human review.

You will receive a batch of model-generated PopQA answers. Select the items most likely to be factually wrong, missing, malformed, or otherwise worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the question and the model answer.
- Do not use or assume the gold answer.
- The desired answer is a short factual answer: an entity, title, place, date, number, or short phrase.
- Prefer examples with likely factual errors, missing answers, multiple conflicting answers, unsupported hedging, non-short-answer formatting, or responses that look too vague to exactly match a valid PopQA answer alias.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
""",
    "mmlupro": """You are selecting examples for human review.

You will receive a batch of model-generated MMLU-Pro answers. Each item is a multiple-choice academic or professional reasoning question with several options. Select the items whose model answers are most likely to be wrong, invalid, ambiguous, or otherwise worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the question, answer options, and model answer.
- Do not use or assume the gold answer.
- The desired answer is a single option letter such as A, B, C, ..., or J.
- Prefer examples with questionable reasoning, answers that name no valid option, multiple conflicting options, unsupported certainty, or a final answer that appears inconsistent with the question and options.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
""",
    "gpqa": """You are selecting examples for human review.

You will receive a batch of model-generated GPQA answers. Each item is a graduate-level science multiple-choice question with four options. Select the items whose model answers are most likely to be wrong, invalid, ambiguous, or otherwise worth sending to a human corrector.

Rules:
- Respect the total budget {budget}. Each item has a listed cost.
- Select any subset of items whose total cost is at most the budget.
- Use only the question, answer options, and model answer.
- Do not use or assume the gold answer.
- The desired answer is a single option letter: A, B, C, or D.
- Prefer examples with questionable scientific reasoning, answers that name no valid option, multiple conflicting options, unsupported certainty, or a final answer that appears inconsistent with the question and options.
- You may select zero items if none appear worth human correction.
- Return only valid JSON in exactly this format:
  {{"selected_indices": [0, 3, 4]}}
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Batch:
{items}
""",
}
