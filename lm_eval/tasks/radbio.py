""""
RadBio AI project - evaluating GPT on our dataset
"""

from lm_eval.base import Task, rf
from pathlib import Path
import pickle
from lm_eval.metrics import mean, f1_score
from best_download import download_file
from zipfile import ZipFile
import os


class RadBio(Task):
    """Base Class for yes/no questions"""

    DATASET_PATH = Path("./radbio_data")

    def download(self, *args, **kwargs):
        if self.DATASET_PATH.exists():
            # don't re-download the dataset
            return
        Path.mkdir(self.DATASET_PATH, parents=True)
        url = "https://docs.google.com/uc?export=download&id=1o2LMR5xdNlTcj2qpSlWEwLPseJxm4AXU&confirm=t"
        checksum = "c78104ee5aaff4339ed9bd30526a01063eccb89765c9842c53de8f10b8accb32"
        zip_path = self.DATASET_PATH / "radbio_question_sets.zip"
        download_file(url, local_file=str(zip_path), expected_checksum=checksum)
        with ZipFile(zip_path, "r") as zip:
            zip.extractall(self.DATASET_PATH)
        os.remove(zip_path)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        if doc["answer"].strip() == "yes":
            gold = 1
        else:
            gold = 0
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class isInSystemQA(RadBio):
    VERSION = 1.0

    def training_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/isInSystemQA/train.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/isInSystemQA/test.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError


class goAHumanQA(RadBio):
    VERSION = 1.0

    def training_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/goAHumanQA/train.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/goAHumanQA/test.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError


class goARadiationResponseQA(RadBio):
    VERSION = 1.0

    def training_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/goARadiationResponseQA/train.pkl",
            "rb",
        ) as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/goARadiationResponseQA/test.pkl",
            "rb",
        ) as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError


class ppiHumanQA(RadBio):
    """Protein protein interaction dataset"""

    VERSION = 1.0

    def training_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/ppiHumanQA/train.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/ppiHumanQA/test.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError


class humanPathwaysQA(RadBio):
    VERSION = 1.0

    def training_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/humanPathwaysQA/train.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def validation_docs(self):
        with open(
            self.DATASET_PATH / "radbio_question_sets/humanPathwaysQA/test.pkl", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    def test_docs(self):
        return NotImplementedError
