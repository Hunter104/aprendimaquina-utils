import os
import io
import inspect
import pandas as pd
import requests
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Union, Tuple

def get_module_name():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return os.path.splitext(os.path.basename(module.__file__))[0] if module and module.__file__ else "unknown"

def generate_classification_markdown(y_true, y_pred) -> str:
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df.round(2).to_markdown()

def generate_confusion_matrix_plot(y_true, y_pred) -> io.BytesIO:
    matrix = confusion_matrix(y_true, y_pred)
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

def get_cv_results_summary(cv_results: Dict) -> Tuple[str, io.BytesIO]:
    results_df = pd.DataFrame(cv_results)
    
    results_df = results_df.sort_values('rank_test_score')
    
    cv_summary = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    cv_summary = cv_summary.head(5)
    
    plt.figure(figsize=(12, 6))
    
    score_cols = [col for col in results_df.columns if col.startswith('split') and col.endswith('_test_score')]
    
    plt.boxplot([results_df[col].values for col in score_cols])
    plt.title('Cross-validation Score Distribution')
    plt.xlabel('CV Fold')
    plt.ylabel('Score')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    
    return cv_summary.to_markdown(), buf

def send_report(
    predictions: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    time_taken: float = None,
    cv_results: Optional[Dict] = None,
    graphs: Optional[List[io.BytesIO]] = None,
    extra_info: Optional[Union[str, Dict]] = None
):
    webhook_url = os.environ.get("WEBHOOK_URL")
    if webhook_url is None:
        raise ValueError("WEBHOOK_URL environment variable not set")
    
    module_name = get_module_name()
    all_graphs = [] if graphs is None else list(graphs)
    
    classification_report_md = ""
    confusion_matrix_buf = None
    cv_summary_md = ""
    cv_summary_buf = None
    
    if predictions is not None:
        if isinstance(predictions, tuple) and len(predictions) == 2:
            y_true, y_pred = predictions
            classification_report_md = generate_classification_markdown(y_true, y_pred)
            confusion_matrix_buf = generate_confusion_matrix_plot(y_true, y_pred)
            all_graphs.append(confusion_matrix_buf)
        else:
            raise ValueError("predictions must be a tuple of (y_true, y_pred)")
    
    if cv_results is not None:
        cv_summary_md, cv_summary_buf = get_cv_results_summary(cv_results)
        all_graphs.append(cv_summary_buf)
    
    extra_info_str = ""
    if extra_info is not None:
        if isinstance(extra_info, dict):
            extra_info_str = "### Model Parameters\n" + "\n".join([f"- **{k}**: {v}" for k, v in extra_info.items()])
        else:
            extra_info_str = str(extra_info)
    
    report_sections = []
    
    report_sections.append("# Training Report")
    
    if time_taken is not None:
        report_sections.append(f"**Time taken:** {time_taken:.2f} s")
    
    if classification_report_md:
        report_sections.append("## Classification Report")
        report_sections.append(classification_report_md)
    
    if cv_results is not None:
        report_sections.append("## Cross-Validation Results")
        report_sections.append("Top performing parameter combinations:")
        report_sections.append(cv_summary_md)
    
    if extra_info_str:
        report_sections.append("## Additional Information")
        report_sections.append(extra_info_str)
    
    report_text = "\n\n".join(report_sections)
    report_buf = io.BytesIO(report_text.encode("utf-8"))
    
    send_file_to_webhook(webhook_url, f"{module_name}_training_report.md", report_buf, "text/markdown")
    
    for i, graph in enumerate(all_graphs):
        send_file_to_webhook(webhook_url, f"{module_name}_graph_{i+1}.png", graph, "image/png")
