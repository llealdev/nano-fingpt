import warnings
from collections import defaultdict

import hdbscan
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_cluster_size: int = 3,
        orphan_cluster_size: int = 2,
        max_tokens: int = 300,
    ):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.min_cluster_size = min_cluster_size
        self.orphan_cluster_sizer = orphan_cluster_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_chunks(self, text_content: str):
        paragrahs = [
            p.strip() for p in text_content.split("\n") if len(p.strip().split()) > 10
        ]

        if not paragrahs:
            return []

        embeddings = self.model.encode(paragrahs, show_progress_bar=False)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
        ).fit_predict(embeddings)

        clusters = defaultdict(list)
        orphans = []
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(paragrahs[i])
            else:
                orphans.append(paragrahs[i])

        final_chunks = []
        for cluster_paras in clusters.values():
            current_chunk = []
            current_tokens = 0

            for para in cluster_paras:
                para_tokens = len(
                    self.tokenizer.encode(para, add_special_tokens=False),
                )

                if current_tokens + para_tokens > self.max_tokens and current_chunk:
                    final_chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens

                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

            if current_chunk:
                final_chunks.append("\n\n".join(current_chunk))

        if len(orphans) > 1:
            orphan_emb = self.model.encode(orphans, show_progress_bar=False)
            orphan_labels = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric="euclidean",
            ).fit_predict(orphan_emb)

            orphan_clusters = defaultdict(list)
            single_orphans = []

            for i, lbl in enumerate(orphan_labels):
                if lbl != -1:
                    orphan_clusters[lbl].append(orphans[i])
                else:
                    single_orphans.append(orphans[i])

            for orphans_paras in orphan_clusters.values():
                current_chunk = []
                current_tokens = 0

                for para in orphans_paras:
                    para_tokens = len(
                        self.tokenizer.encode(para, add_special_token=False)
                    )

                    if current_tokens + para_tokens > self.max_tokens and current_chunk:
                        final_chunks.append("\n\n".join(current_chunk))
                        current_chunk = [para]
                        current_tokens = para_tokens
                    else:
                        current_chunk.append(para)
                        current_tokens += para_tokens

                if current_chunk:
                    final_chunks.append("\n\n".join(current_chunk))

            final_chunks.extend(single_orphans)

        elif orphans:
            final_chunks.append(orphans[0])

        return final_chunks
