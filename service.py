import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classfier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(),output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    result =iris_clf_runner.predict.run(input_series)
    return result
