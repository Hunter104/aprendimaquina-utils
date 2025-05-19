import os
import io
import inspect
import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt

webhook_url = os.environ.get("WEBHOOK_URL")
if webhook_url is None:
    raise ValueError("WEBHOOK_URL environment variable not set")

def send_report(
    y_test, y_pred, time_taken: str,
    graphs: Optional[list] = None,
    extra_info: Optional[str] = None
):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    module_name = os.path.splitext(os.path.basename(module.__file__))[0] if module and module.__file__ else "unknown"

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_md = report_df.round(2).to_markdown()

    matrix = confusion_matrix(y_test, y_pred)
    matrix_index = pd.Index([f"Real {i}" for i in range(matrix.shape[0])])
    matrix_columns = pd.Index([f"Pred {i}" for i in range(matrix.shape[1])])
    matrix_df = pd.DataFrame(matrix, index=matrix_index, columns=matrix_columns)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    graph_buf = io.BytesIO()
    plt.savefig(graph_buf, format="png")
    plt.close()
    graph_buf.seek(0)
    graphs = graphs or []
    graphs.append(graph_buf)

    report = f"""\
# Training Report
**Time taken:** {time_taken} s

## Classification Report
{report_md}

## Observations
{extra_info if extra_info else "No additional observations."}
"""

    buffer = io.BytesIO()
    buffer.write(report.encode("utf-8"))
    buffer.seek(0)

    report_filename = f"{module_name}_training_report.md"
    requests.post(webhook_url, files={'file': (report_filename, buffer, "text/markdown")})

    if graphs:
        for i, graph_buf in enumerate(graphs):
            graph_buf.seek(0)
            filename = f"{module_name}_graph_{i+1}.png"
            requests.post(webhook_url, files={'file': (filename, graph_buf, "image/png")})
