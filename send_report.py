import os
import io
import inspect
import textwrap
import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List


def get_module_name():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return os.path.splitext(os.path.basename(module.__file__))[0] if module and module.__file__ else "unknown"


def generate_classification_markdown(y_test, y_pred) -> str:
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df.round(2).to_markdown()


def generate_confusion_matrix_plot(y_test, y_pred) -> io.BytesIO:
    matrix = confusion_matrix(y_test, y_pred)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Real {i}" for i in range(matrix.shape[0])],
        columns=[f"Pred {i}" for i in range(matrix.shape[1])],
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


def send_file_to_webhook(webhook_url: str, filename: str, file_buffer: io.BytesIO, mime_type: str):
    file_buffer.seek(0)
    requests.post(webhook_url, files={'file': (filename, file_buffer, mime_type)})


def send_report(
    y_test,
    y_pred,
    time_taken: str,
    graphs: Optional[List[io.BytesIO]] = None,
    extra_info: Optional[str] = None
):
    webhook_url = os.environ.get("WEBHOOK_URL")
    if webhook_url is None:
        raise ValueError("WEBHOOK_URL environment variable not set")

    module_name = get_module_name()
    report_md = generate_classification_markdown(y_test, y_pred)
    main_graph = generate_confusion_matrix_plot(y_test, y_pred)

    report_text = textwrap.dedent(f"""\
        # Training Report
        **Time taken:** {time_taken} s

        ## Classification Report
        {report_md}

        ## Observations
        {extra_info or "No additional observations."}
    """)
    report_buf = io.BytesIO(report_text.encode("utf-8"))
    send_file_to_webhook(webhook_url, f"{module_name}_training_report.md", report_buf, "text/markdown")

    all_graphs = [main_graph] + (graphs or [])
    for i, graph in enumerate(all_graphs):
        send_file_to_webhook(webhook_url, f"{module_name}_graph_{i+1}.png", graph, "image/png")
