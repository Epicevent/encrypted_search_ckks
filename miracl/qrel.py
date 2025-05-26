TARGET_QID = "2"
import ir_datasets
dataset = ir_datasets.load("miracl/ko/dev")
for qrel in dataset.qrels_iter():
    if str(qrel.query_id) == TARGET_QID:
        print(f"  • doc_id={qrel.doc_id}, relevance={qrel.relevance}")
