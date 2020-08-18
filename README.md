# Questionnaire Generator Models

A basic API for exposing pre-trained models related to the task of generation of questionaries,
to be more specific, for generating multiple choice questions (MCQs).

This models can be used by other APIs so they can combine these model predictions for implementing
different solutions related to the generation of questionnaires (i.e question generation, question
answering, distractors generation, etc.)

## Installation
This project uses [Cortex](https://www.cortex.dev/) for running and deployment, so it's as simple as install it and deploy (locally or AWS).

```
# Install cortex
bash -c "$(curl -sS https://raw.githubusercontent.com/cortexlabs/cortex/0.18/get-cli.sh)"

# Clone this repo:
git clone https://github.com/rrodrigu3z/questionnaire-generator-api.git

# Deploy predictors
cd questionnaire-generator-api/predictors
cortex deploy
```

## Models
Here's the list of models exposed:

- **valhalla/t5-base-qg-hl:** Pretrained T5 model in the task of QG / QA.
- **gpt2:** Pretrained GPT2 model for text generation, can be used for distractors generation.
- **t5-distractor:** T5 Model I fine-tuned for distractor generation.
- **ramsrigouthamg/t5_paraphraser:** T5 Pretrained model in the task of paraphrasing questions.

Check `cortex.yml` for details about the config for each model, like port they are exposed and
decoding params used for generating results.

## Examples:

### Question generation
```
cx predict t5-question-generation ../samples/article_sample.json
```

```json
{
  "result": [
    {
      "answer": "reasoning, knowledge representation, planning, learning, natural language processing, perception and the ability to move and manipulate objects",
      "question": "What are the traditional goals of AI research?"
    },
    {
      "answer": "General intelligence",
      "question": "What is one of the long-term goals of AI?"
    },
    {
      "answer": "statistical methods, computational intelligence, and traditional symbolic AI",
      "question": "What approaches are used in AI research?"
    },
    {
      "answer": "artificial neural networks",
      "question": "What is an example of a tool used in AI?"
    },
    {
      "answer": "computer science, information engineering, mathematics, psychology, linguistics, philosophy, and many other fields",
      "question": "What does the AI field draw upon?"
    }
  ],
  "took": 118.72984385490417
}
```

### Distractor generation (T5)
```
cx predict t5-distractor-generation ../samples/distractor_generation_sample.json
```

```json
{
    "result": [
        "Artificial neural networks",
        "Social algorithms",
        "Understanding AI",
        "Artificial neural networks and conventional symbolic AI",
        "AI",
        "Modern data",
        "Reasoning, knowledge representation, planning and learning",
        "Cognitive intelligence",
        "Astronomical algorithms"
    ],
    "took": 11.18682861328125
}
```

### Creative distractor generation (GPT2)
```
cx predict gpt2-text-generation ../samples/text_generation_sample.json
```

```json
{
    "result": [
        {
            "generated_text": "Q: What is one of the long-term goals of AI?\nA: We don't know yet, but it's something we're looking at. We're going to have to see what we can do to improve it. It's going"
        },
        {
            "generated_text": "Q: What is one of the long-term goals of AI?\nA: The goal is to make AI more efficient. It's a good idea to think about AI as a sort of high-level language. For example, we want to"
        },
        {
            "generated_text": "Q: What is one of the long-term goals of AI?\nA: We want to be able to predict the behavior of other species. We don't want it to become an AI. It's a human-like system.\nQ"
        }
    ],
    "took": 32.06318020820618
}
```

## Deploy to AWS
...
