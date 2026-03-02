"""Dataset transformation functions for Axolotl preprocessing."""


def arc_easy_to_messages(sample):
    """Transform ARC-Easy format to messages format.

    Converts allenai/ai2_arc samples with multiple-choice structure
    into user/assistant message pairs suitable for chat_template.

    Args:
        sample: Dict with keys 'question', 'choices', 'answerKey'

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    choices_text = "\n".join(
        [
            f"{label}: {text}"
            for label, text in zip(
                sample["choices"]["label"], sample["choices"]["text"], strict=False
            )
        ]
    )

    # Build prompt matching exps-mom template
    user_message = (
        "Answer the question below by choosing the correct choice.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Choices: {choices_text}\n\n"
        "You must respond with the letter corresponding to the correct "
        "choice (A,B,C,D) without any explanation. Answer:"
    )

    assistant_message = sample["answerKey"]

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def piqa_to_messages(sample):
    """Transform PIQA format to messages format.

    Converts ybisk/piqa samples with physical commonsense reasoning
    into user/assistant message pairs suitable for chat_template.

    Args:
        sample: Dict with keys 'goal', 'sol1', 'sol2', 'label'
            - goal: str (the physical commonsense question)
            - sol1: str (first solution)
            - sol2: str (second solution)
            - label: int (0 for sol1, 1 for sol2)

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    choices_text = f"A: {sample['sol1']}\nB: {sample['sol2']}"

    user_message = (
        "Answer the question below by choosing the correct choice.\n\n"
        f"Goal: {sample['goal']}\n\n"
        f"Choices: {choices_text}\n\n"
        "You must respond with the letter corresponding to the correct "
        "choice (A,B) without any explanation. Answer:"
    )

    # Convert 0/1 label to A/B
    assistant_message = "A" if sample["label"] == 0 else "B"

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def hellaswag_to_messages(sample):
    """Transform HellaSwag format to messages format.

    Converts Rowan/hellaswag samples with commonsense NLI
    into user/assistant message pairs suitable for chat_template.

    Args:
        sample: Dict with keys 'ctx', 'endings', 'label'
            - ctx: str (context/premise)
            - endings: list of str (possible continuations)
            - label: str or int (correct ending index)

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        [f"{labels[i]}: {ending}" for i, ending in enumerate(sample["endings"])]
    )

    user_message = (
        "Complete the context below by choosing the correct continuation.\n\n"
        f"Context: {sample['ctx']}\n\n"
        f"Choices: {choices_text}\n\n"
        "You must respond with the letter corresponding to the correct "
        "choice (A,B,C,D) without any explanation. Answer:"
    )

    # Convert label to letter (handle both int and string labels)
    # Test splits may have empty labels — produce an empty assistant message
    # so the sample can be filtered out downstream.
    raw_label = sample["label"]
    if isinstance(raw_label, str) and raw_label == "":
        assistant_message = ""
    else:
        label_idx = int(raw_label) if isinstance(raw_label, str) else raw_label
        assistant_message = labels[label_idx]

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def openbookqa_to_messages(sample):
    """Transform OpenBookQA format to messages format.

    Converts allenai/openbookqa samples with elementary science questions
    into user/assistant message pairs suitable for chat_template.

    Args:
        sample: Dict with keys 'question_stem', 'choices', 'answerKey'
            - question_stem: str (the question)
            - choices: dict with 'text' (list) and 'label' (list)
            - answerKey: str (correct answer letter)

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    choices_text = "\n".join(
        [
            f"{label}: {text}"
            for label, text in zip(
                sample["choices"]["label"], sample["choices"]["text"], strict=False
            )
        ]
    )

    user_message = (
        "Complete the following passage or answer the question by choosing the correct choice.\n\n"
        f"Question: {sample['question_stem']}\n\n"
        f"Choices: {choices_text}\n\n"
        "You must respond with the letter corresponding to the correct "
        "choice (A,B,C,D) without any explanation. Answer:"
    )

    assistant_message = sample["answerKey"]

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def winogrande_to_messages(sample):
    """Transform Winogrande format to messages format.

    Converts winogrande samples with commonsense reasoning about pronoun resolution.

    Args:
        sample: Dict with keys 'sentence', 'option1', 'option2', 'answer'
            - sentence: str (sentence with blank _)
            - option1: str (first option to fill blank)
            - option2: str (second option to fill blank)
            - answer: str ("1" or "2")

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    user_message = (
        "Answer the question below by choosing the correct choice.\n\n"
        f"Question: {sample['sentence']}\n\n"
        f"Choices: 0: {sample['option1']}\n1: {sample['option2']}\n\n"
        "You must respond with the number corresponding to the correct choice (0, 1) without any explanation."
    )

    # Convert 1,2 labels to 0,1
    # Test splits may have empty answers — produce an empty assistant message
    # so the sample can be filtered out downstream.
    raw_answer = sample["answer"]
    if isinstance(raw_answer, str) and raw_answer == "":
        assistant_message = ""
    else:
        assistant_message = str(int(raw_answer) - 1)

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def social_iqa_to_messages(sample):
    """Transform Social IQA format to messages format.

    Converts social_i_qa samples with social commonsense reasoning.

    Args:
        sample: Dict with keys 'context', 'question', 'answerA', 'answerB', 'answerC', 'label'
            - context: str (social situation)
            - question: str (question about the situation)
            - answerA/B/C: str (possible answers)
            - label: str ("1", "2", or "3")

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    choices_text = (
        f"1: {sample['answerA']}\n2: {sample['answerB']}\n3: {sample['answerC']}"
    )

    user_message = (
        "Answer the question below by choosing the correct choice.\n\n"
        f"Context: {sample['context']}\n\n"
        f"Question: {sample['question']}\n\n"
        f"Choices: {choices_text}\n\n"
        "You must respond with the number corresponding to the correct choice (1, 2, 3) without any explanation. Answer:"
    )

    assistant_message = sample["label"]

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def math_qa_to_messages(sample):
    """Transform MathQA format to messages format.

    Converts allenai/math_qa samples with math word problems and multiple choices.

    Args:
        sample: Dict with keys 'Problem', 'options', 'correct'
            - Problem: str (math problem statement)
            - options: str (formatted options string)
            - correct: str (correct option letter)

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    user_message = (
        "Answer the question below by choosing the correct choice.\n\n"
        f"Question: {sample['Problem']}\n\n"
        f"Choices: {sample['options']}\n\n"
        "You must respond with the letter corresponding to the correct choice (a, b, c, d, e) without any explanation."
    )

    assistant_message = sample["correct"]

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def commonsense_qa_to_messages(sample):
    """Transform CommonsenseQA format to messages format.

    Converts tau/commonsense_qa samples with commonsense reasoning questions.

    Args:
        sample: Dict with keys 'question', 'choices', 'answerKey'
            - question: str (commonsense question)
            - choices: dict with 'text' (list) and 'label' (list)
            - answerKey: str (correct answer letter A-E)

    Returns:
        Dict with 'messages' list containing user/assistant turns
    """
    choices_text = "\n".join(
        [
            f"{label}: {text}"
            for label, text in zip(
                sample["choices"]["label"], sample["choices"]["text"], strict=False
            )
        ]
    )

    user_message = (
        "Answer the question below by choosing the correct choice.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Choices: {choices_text}\n\n"
        "You must respond with the letter corresponding to the correct choice (A,B,C,D,E) without any explanation. Answer:"
    )

    assistant_message = sample["answerKey"]

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }
