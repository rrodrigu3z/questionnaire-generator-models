# initialization code and variables can be declared here in global scope
import nltk
from api_response import response
from question_generation.pipelines import pipeline

class PythonPredictor:
    def __init__(self, config):
        """Called once before the API becomes available. Performs setup such as downloading/initializing the model or downloading a vocabulary.

        Args:
            config: Dictionary passed from API configuration (if specified). This may contain information on where to download the model and/or metadata.
        """
        nltk.download('punkt')
        self.nlp = pipeline(config["pipeline"], model=config["model"])

    @response
    def predict(self, payload, query_params, headers):
        """Called once per request. Preprocesses the request payload (if necessary), runs inference, and postprocesses the inference output (if necessary).

        Args:
            payload: The request payload (see below for the possible payload types) (optional).
            query_params: A dictionary of the query parameters used in the request (optional).
            headers: A dictionary of the headers sent in the request (optional).

        Returns:
            Prediction or a batch of predictions.
        """
        return self.nlp(payload["paragraph"])
