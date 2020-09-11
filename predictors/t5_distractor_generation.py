# initialization code and variables can be declared here in global scope
import nltk
import spacy
import en_core_web_lg
from api_response import response
from t5_model.pipelines import pipeline as t5_pipeline
from utils.downloads import download_file
from utils.stop_words import remove_stop_words
from os import path, makedirs


MODEL_CONFIG_FILE = "config.json"
MODEL_BIN_FILE = "pytorch_model.bin"


class PythonPredictor:
    def __init__(self, config):
        """Called once before the API becomes available.
        Downloads models, requirements and initializes supported pipelines:

        - Distractor Generation: generates distractors for MCAQs.

        Args:
            config: Dictionary passed from API configuration (if specified).
                    Contains info about models to use and params.
        """
        self.config = config
        self._download_model()
        self.nlp = en_core_web_lg.load()

        self.distractor_generation = t5_pipeline(
            config["pipeline"],
            model=config["model"],
            base_model=config["base_model"])

    @response
    def predict(self, payload, query_params, headers):
        """Called once per request. Preprocesses the request payload (if necessary),
        runs inference, and postprocesses the inference output (if necessary).

        Args:
            payload: The request payload (see below for the possible payload types) (optional).
            query_params: A dictionary of the query parameters used in the request (optional).
            headers: A dictionary of the headers sent in the request (optional).

        Returns:
            Prediction or a batch of predictions.
        """
        # Get prediction decoding and search params and
        # support overrides from the payload.
        decoding = {
            **self.config["decoding_params"],
            **payload.get("decoding", {})
        }

        distractors = self.distractor_generation(
            payload["question"], payload["answer"], payload["context"],
            decoding)

        return self._rank(
            distractors,
            self.nlp(remove_stop_words(payload["answer"])))

    def _rank(self, distractors, nlp_answer):
        """Adds a similarity score to each distractor"""
        def add_rank(item):
            nlp_distractor = self.nlp(remove_stop_words(item["distractor"]))
            item["similarity"] = nlp_answer.similarity(nlp_distractor)
            return item
        return list(map(add_rank, distractors))

    def _download_model(self):
        nltk.download("punkt")
        nltk.download('stopwords')

        if self.config["requires_download"]:
            makedirs(self.config["model"], exist_ok=True)
            self._download_model_file(MODEL_CONFIG_FILE)
            self._download_model_file(MODEL_BIN_FILE)

    def _download_model_file(self, filename):
        output_file = path.join(self.config["model"], filename)

        if not path.isfile(output_file):
            url = path.join(self.config["download_path"], filename)
            download_file(url=url, filename=output_file, verbose=True)
