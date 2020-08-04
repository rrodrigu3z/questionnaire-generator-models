import time
import logging

def response(api_predict):
    def _log(key, data):
        logging.info(f"{key}:\n{data}")
    def predict(self, payload, query_params, headers):
        _log("PAYLOAD", payload)
        _log("QUERY PARAMS", query_params)
        _log("HEADERS", headers)

        start = time.time()
        response = api_predict(self, payload, query_params, headers)
        took = time.time() - start

        return {"result": response, "took": took}
    return predict
