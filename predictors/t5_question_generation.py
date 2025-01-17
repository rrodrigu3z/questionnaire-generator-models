# initialization code and variables can be declared here in global scope
import nltk
from api_response import response
from question_generation.pipelines import pipeline


class PythonPredictor:
    def __init__(self, config):
        """Called once before the API becomes available.
        Downloads models, requirements and initializes supported pipelines:

        - Question Generation: generates question and answers
        - Question Paraphrasing: paraphrases a given question.

        Args:
            config: Dictionary passed from API configuration (if specified).
                    Contains info about models to use and params.
        """
        nltk.download("punkt")

        self.question_generation = pipeline(
            config["pipeline"], model=config["model"])

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
        # Generates question & answers for a given paragraph
        return self.question_generation(payload["paragraph"])
