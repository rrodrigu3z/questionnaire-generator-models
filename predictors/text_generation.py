# initialization code and variables can be declared here in global scope
from api_response import response
from transformers import pipeline, GPT2Tokenizer

class PythonPredictor:
    def __init__(self, config):
        """Called once before the API becomes available. Performs setup such as downloading/initializing the model or downloading a vocabulary.

        Args:
            config: Dictionary passed from API configuration (if specified). This may contain information on where to download the model and/or metadata.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(config["model"])
        self.generator = pipeline(config["pipeline"], model=config["model"])
        self.decoding_params = config["decoding_params"]
        self.decoding_params["eos_token_id"] = tokenizer.eos_token_id

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
        decoding = {**self.decoding_params, **payload.get("decoding", {})}
        return self.generator(payload["context"], **decoding)
