from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import ResponseBody, BatchTextResponse, TextResponse
from flask_ml.flask_ml_server.templates.utils import response_body
from flask_ml.flask_ml_server.templates.batchfile import FileInputs, create_task_schema_func
from flask_ml.flask_ml_server.templates.no_parameters import NoParameters

from ..ml.model import AudioTranscriptionModel

model = AudioTranscriptionModel()
server = MLServer(__name__)


@server.route("/transcribe", task_schema_func=create_task_schema_func())
def transcribe(inputs: FileInputs, parameters: NoParameters) -> ResponseBody:
    print("Inputs:", inputs)
    print("Parameters:", parameters)
    files = [e.path for e in inputs["file_inputs"].files]
    results = model.transcribe_batch(files)
    results = {r["file_path"]: r["result"] for r in results}
    return response_body(
        BatchTextResponse(
            texts=[
                TextResponse(
                    title=k,
                    value=v,
                )
                for k, v in results.items()
            ]
        )
    )

server.run()
