"""
Paraphrasing pipelines

Based on work:
https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5GenerationPipeline:
    """Abstract class for conditional text generation
    using a pre-trainded T5 model
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, decoding_params):
        raise NotImplementedError

    def _generate(self, input_ids, attention_masks, decoding_params):
        params = {**decoding_params,
                  "input_ids": input_ids,
                  "attention_mask": attention_masks}

        beam_outputs = self.model.generate(**params)
        return beam_outputs

    def _decode(self, beam_outputs, original_text):
        final_outputs = []
        lower_outputs = []

        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)
            if (sent.lower() != original_text.lower() and
                    sent.lower() not in lower_outputs):
                # Only add the decoded text if not the same as the question
                # and is not duplicated
                final_outputs.append(sent)
                lower_outputs.append(sent.lower())

        return final_outputs


class QuestionParaphraserPipeline(T5GenerationPipeline):
    """Implementation of a pipeline for question paraphrasing
    using a pre-trained T5 model.
    """
    def __call__(self, question, decoding_params):
        encoding = self.tokenizer.encode_plus(
            f"paraphrase: {question} </s>",
            pad_to_max_length=True,
            return_tensors="pt")

        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        beam_outputs = self._generate(
            input_ids, attention_masks, decoding_params)

        return self._decode(beam_outputs, question)


class DistractorGenerationPipeline(T5GenerationPipeline):
    """Implementation of a pipeline for generation of distractors
    to be used in MCQs using a pre-trained T5 model.
    """
    def __call__(self, question, answer, context, decoding_params):
        text = (f"generate distractor: {question}  "
                f"answer: {answer} "
                f"context: {context} </s>")

        encoding = self.tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt")

        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        beam_outputs = self._generate(
            input_ids, attention_masks, decoding_params)

        return self._decode(beam_outputs, answer)


SUPPORTED_PIPELINES = {
    "question-paraphrase": {
        "klass": QuestionParaphraserPipeline
    },
    "distractor-generation": {
        "klass": DistractorGenerationPipeline
    }
}


def pipeline(name, model, base_model):
    model = T5ForConditionalGeneration.from_pretrained(model)
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pipeline_config = SUPPORTED_PIPELINES[name]
    klass = pipeline_config["klass"]

    return klass(model, tokenizer, device)

