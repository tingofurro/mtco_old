from tasks.summary.eval_summhay import build_ref_insight2docids, compute_single_sample_results, evaluate_insights
from collections import Counter
import os, json, random, math
from task_base import Task

# random.seed(42)

def download_summhay():
    for domain in ["news", "conv"]:
        for i in range(1,6):
            # https://raw.githubusercontent.com/salesforce/summary-of-a-haystack/refs/heads/master/data/topic_conv1.json
            # save it as data/summhay/topic_{domain}{i}.json
            url = f"https://raw.githubusercontent.com/salesforce/summary-of-a-haystack/refs/heads/master/data/topic_{domain}{i}.json"
            if not os.path.exists(f"data/summhay/topic_{domain}{i}.json"):
                print(f"Downloading {url} to data/summhay/topic_{domain}{i}.json")
                os.system(f"wget {url} -O data/summhay/topic_{domain}{i}.json")
                # on windows rather than unix
                # os.system(f"powershell -Command \"Invoke-WebRequest -Uri {url} -OutFile data/summhay/topic_{domain}{i}.json\"")


def generate_sharded_summhay_samples():
    download_summhay()

    samples = []
    for fn in os.listdir("data/summhay"):
        with open(f"data/summhay/{fn}", "r") as f:
            topic = json.load(f)
        domain = "news" if "news" in fn else "conv"

        for subtopic in topic["subtopics"]:
            query = subtopic["query"]
            oracle_scores = subtopic["retriever"]["oracle"]
            insights = subtopic["insights"]
            insight_ids = [insight["insight_id"] for insight in insights]

            documents = topic["documents"]
            real_doc_ids = set([doc["document_id"] for doc in documents])

            selected_doc_ids = []
            insight_id_counts = Counter()

            num_repeats = 2

            while any([insight_id_counts[insight_id] < num_repeats for insight_id in insight_ids]):
                random.shuffle(documents)
                left_insight_ids = set([insight_id for insight_id in insight_ids if insight_id_counts[insight_id] < num_repeats])
                doc_sorted = sorted(documents, key=lambda x: (len(set(x["insights_included"]) & left_insight_ids) * (0 if x["document_id"] in selected_doc_ids else 1)), reverse=True)
                selected_doc = doc_sorted[0]
                selected_doc_ids.append(selected_doc["document_id"])
                insight_id_counts.update(selected_doc["insights_included"])

            distractor_doc_ids = [id for id, score in oracle_scores.items() if score == 0 and id in real_doc_ids]
            distractor_doc_ids = random.sample(distractor_doc_ids, max(len(selected_doc_ids), 6))
            selected_doc_ids = distractor_doc_ids + selected_doc_ids

            for doc_idx, doc in enumerate(topic["documents"]):
                doc["document_index"] = doc_idx+1

            docs = [doc for doc in topic["documents"] if doc["document_id"] in selected_doc_ids]

            doc_keep_keys = ["document_id", "document_index", "document_text", "insights_included"]
            docs = [{k: doc[k] for k in doc_keep_keys} for doc in docs]

            final_doc_idxs = [doc["document_index"] for doc in docs]
            assert sorted(final_doc_idxs) == final_doc_idxs, "Document indices are not sorted"
            N_turns = 10

            # split the doc ids into N_turns
            random.shuffle(final_doc_idxs)
            per_batch = math.ceil(len(final_doc_idxs) / N_turns)
            # print(N_turns, per_batch)
            doc_idxs_batches = [final_doc_idxs[i:i+per_batch] for i in range(0, len(final_doc_idxs), per_batch)]
            insightid2ref_citations = build_ref_insight2docids(topic)

            sample = {
                "topic_id": topic["topic_id"],
                "topic": topic["topic"],
                "subtopic_id": subtopic["subtopic_id"],
                "domain": domain,
                "query": query,
                "documents": docs,
                "insights": subtopic["insights"],
                "insightid2ref_citations": insightid2ref_citations,
                "shards": [
                    {"shard_id": i, "shard": "", "doc_idxs": doc_idxs_batches[i]} for i in range(len(doc_idxs_batches))
                ]
            }
            samples.append(sample)
            # return

    for i, sample in enumerate(samples):
        sample["task_id"] = f"sharded_summary_{i}"
        sample["task"] = "summary"
        sample["original_task_id"] = f"{sample['topic_id']}_{sample['subtopic_id']}"

    random.shuffle(samples)

    with open(f"data/sharded_summary_tmp.json", "w") as f:
        json.dump(samples, f, indent=4)

class TaskSummary(Task):
    def __init__(self):
        # self.version = version
        with open(f"prompts/summary/summary_full_prompt_conv.txt", "r") as f:
            self.fully_specified_prompt_conv = f.read()
        with open(f"prompts/summary/summary_full_prompt_news.txt", "r") as f:
            self.fully_specified_prompt_news = f.read()
        with open(f"prompts/summary/summary_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        self.answer_extraction_strategy = "full_response"

    def get_answer_description(self) -> str:
        return "A complete summary potentially containing multiple lines, and citation."

    def generate_system_prompt(self, sample):
        return self.system_prompt

    def get_task_name(self) -> str:
        return "summary"

    def get_dataset_file(self) -> str:
        return "data/sharded_instructions_600.json"

    def get_samples(self):
        with open(self.get_dataset_file(), "r") as f:
            samples = json.load(f)
        samples = [d for d in samples if d["task"] == "summary"]
        return samples


    def evaluator_function(self, extracted_answer, sample):
        evaluator_model_card = "t-gpt-4o" if os.environ.get("USE_TRAPI", "0") == "1" else "gpt-4o"
        evals = evaluate_insights(sample["insights"], extracted_answer, evaluator_model_card, "prompts/summary/summhay_evaluation.txt")
        # eval should likely be cached somewhere, so results can be explained if needed
        results = compute_single_sample_results(extracted_answer, evals, sample["insightid2ref_citations"])
        # results["score"] = results["coverage_score"]
        results["score"] = results["joint_score"] # we save this as the main score we anchor on
        return results

    def populate_fully_specific_prompt(self, sample):
        prompt = self.fully_specified_prompt_conv if sample["domain"] == "conv" else self.fully_specified_prompt_news

        documents_txt = ""
        for document in sample["documents"]:
            documents_txt += f"Document {document['document_index']}:\n{document['document_text']}\n\n"

        prompt = prompt.replace("[[TOPIC]]", sample["topic"]).replace("[[DOCUMENTS]]", documents_txt).replace("[[QUERY]]", sample["query"]).replace("[[N_DOCS]]", str(len(sample["documents"]))).replace("[[N_INSIGHTS]]", str(len(sample["insights"])))
        return prompt

    def populate_concat_prompt(self, sample):
        prompt = self.fully_specified_prompt_conv if sample["domain"] == "conv" else self.fully_specified_prompt_news
        documents_txt = "The documents were received in multiple chunks, you can disregard the chunking information, and consider all documents equally."
        doc_idx2doc = {doc["document_index"]: doc["document_text"] for doc in sample["documents"]}

        for i, shard in enumerate(sample["shards"]):
            documents_txt += f"Document Chunk {i+1}:\n"
            for doc_idx in shard["doc_idxs"]:
                documents_txt += f"Document {doc_idx}:\n{doc_idx2doc[doc_idx]}\n\n"

        prompt = prompt.replace("[[TOPIC]]", sample["topic"]).replace("[[DOCUMENTS]]", documents_txt).replace("[[QUERY]]", sample["query"]).replace("[[N_DOCS]]", str(len(sample["documents"]))).replace("[[N_INSIGHTS]]", str(len(sample["insights"])))
        return prompt
    
    def populate_sharded_prompt(self, sample, turn_index):
        doc_idx2doc = {doc["document_index"]: doc["document_text"] for doc in sample["documents"]}
        if turn_index == 0:
            shard = sample["shards"][0]
            prompt = self.fully_specified_prompt_conv if sample["domain"] == "conv" else self.fully_specified_prompt_news
            documents_txt = ""
            for doc_idx in shard["doc_idxs"]:
                documents_txt += f"Document {doc_idx}:\n{doc_idx2doc[doc_idx]}\n\n"
            prompt = prompt.replace("[[TOPIC]]", sample["topic"]).replace("[[DOCUMENTS]]", documents_txt).replace("[[QUERY]]", sample["query"]).replace("[[N_DOCS]]", str(len(sample["documents"]))).replace("[[N_INSIGHTS]]", str(len(sample["insights"])))
            return prompt, shard["shard_id"], 0.0
        elif turn_index <= len(sample["shards"]):
            shard = sample["shards"][(turn_index-1)]
            documents_txt = ""
            for doc_idx in shard["doc_idxs"]:
                documents_txt += f"Document {doc_idx}:\n{doc_idx2doc[doc_idx]}\n\n"
            prompt = f"I have found a few additional documents, please rewrite the summary considering all documents so far (from before, and the new ones).\n\n{documents_txt}"
            return prompt, shard["shard_id"], 0.0
        else:
            return None, -1, 0.0

    
    def process_original_sample(self, sample):
        return {
            "task_id": sample["task_id"],
            "topic": sample["topic"],
            "query": sample["query"],
            "documents": sample["documents"],
            "insights": sample["insights"]
        }

if __name__ == "__main__":
    generate_sharded_summhay_samples()
