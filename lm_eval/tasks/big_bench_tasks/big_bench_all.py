"""
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative
benchmark intended to probe large language models and extrapolate
their future capabilities. The more than 200 tasks included in BIG-bench are
summarized by keyword here: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/keywords_to_tasks.md#summary-table,
and by task name here: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/README.md.
A paper introducing the benchmark, including evaluation results on large language models,
is currently in preparation.

Homepage: https://github.com/google/BIG-bench
"""

# TODO: Figure out how to grade tasks that have both mc and generative portions
# TODO: Get all tasks in

import numpy as np
import datasets
import sacrebleu
from rouge_score import rouge_scorer, scoring

from lm_eval.base import Task, rf
from lm_eval.metrics import mean

_CITATION = """@misc{https://doi.org/10.48550/arxiv
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models},

  publisher = {arXiv},

  year = {2022},

  copyright = {arXiv.org perpetual, non-exclusive license}
}

"""


class BigBench_General(Task):
    VERSION = "0.0.1"
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "bigbench"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        # TODO figure out how to see if they have both mc and gen tasks
        if len(self.dataset["default"][0]["multiple_choice_targets"]) != 0:
            self.MC = True
        else:
            self.MC = False

        # Default metrics, change if needed
        self.metrics = ("absolute_match", "bleurt", "bleu")
        if "bleurt" in self.metrics:
            # Force bleurt to run on CPU, otherwise we will run out of memory on GPU
            # when running gpt-neox
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
            self.bluert = datasets.load_metric(
                "bleurt",
            )

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return map(self._process_doc, self.dataset["default"])

    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.

        # differentiate between multiple choice and generative based on the doc
        # As of now, gen targets do not have any mc targets, so we will go based on that

        # MC targets
        if self.MC:
            query = doc["inputs"]
            choices = doc["multiple_choice_targets"]
            gold = doc["multiple_choice_scores"].index(1)
            return {
                "query": query,  # The query prompt.
                "choices": choices,  # The list of choices.
                "gold": gold,  # The integer used to index into the correct element of `"choices"`.
            }
        # generative target, might be useful for having an elif here, but need a bit more specificity on dataset
        else:
            query = doc["inputs"]
            targets = doc["targets"]
            return {
                "query": query,  # The query prompt.
                "targets": targets,  # the answer(s)
            }

    def doc_to_text(self, doc):
        # default for mc? might not work...
        # Used it in the generative tasks file, hopefully its okay. Looks good on visual inspection
        return doc["query"]

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.

        # Start with MC first
        # if "choices" in doc.keys():
        if self.MC:
            return " " + doc["choices"][doc["gold"]]  # stolen from base.py

        # Generative tasks
        else:
            target = doc["targets"]
            # TODO: what if a list of elems ? for few shot?  This is a hack
            if isinstance(target, list):
                return " " + target[0]
            return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # if "choices" in doc.keys():
        if self.MC:
            lls = [
                rf.loglikelihood(ctx, " {}".format(choice))[0]
                for choice in doc["choices"]
            ]

            return lls
        else:
            # TODO: Construct your language model requests with the request factory, `rf`,
            # and return them as an iterable.
            # stolen from gsmk8k.py
            completion = rf.greedy_until(ctx, ["\n"])
            return completion

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.

        # Start with multiple choice
        # if "gold" in doc.keys():
        if self.MC:
            return self._grade_mc(doc, results)
        # Generative tasks
        # TODO: replcae with something like bleu? or add it? Look at
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py
        # It is the metrics scorer used in BIG-Bench
        else:
            return self._grade_gen(doc, results)

    def _grade_mc(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc_mc": acc,
            "acc_norm": acc_norm,
        }

    def _grade_gen(self, doc, results):
        metrics = {}
        if "absolute_match" in self.metrics:
            targets = doc["targets"]
            gen_answer = results[0]

            gen_answer = gen_answer.lower().strip()
            acc = 0
            if isinstance(targets, list):
                for target in targets:
                    if gen_answer == target.lower().strip():
                        acc = 1
            else:
                target = targets
                if gen_answer == target.lower().strip():
                    acc = 1

            metrics["absolute_match_acc"] = acc

        if "bleurt" in self.metrics:
            gen_answer = results[0].strip()
            targets = doc["targets"]
            if isinstance(targets, str):
                targets = [targets]

            bleurt_scores = self.bluert.compute(
                predictions=[gen_answer] * len(targets), references=targets
            )["scores"]

            bleurt_max = max(bleurt_scores)
            metrics["bleurt"] = bleurt_max

        if "bleu" in self.metrics:
            bleu_scores = [self.bleu([[targ]], [gen_answer]) for targ in targets]
            bleu_max = np.nanmax(bleu_scores)

            metrics["bleu"] = bleu_max

        if "rogue" in self.metrics:
            # ROUGE-N
            rouge_scores = [self.rouge([ref], [gen_answer]) for ref in targets]
            # ROUGE-1
            rouge1_scores = [score["rouge1"] for score in rouge_scores]
            rouge1_max = np.nanmax(rouge1_scores)
            # ROUGE-2
            rouge2_scores = [score["rouge2"] for score in rouge_scores]
            rouge2_max = np.nanmax(rouge2_scores)

            # ROUGE-L
            rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
            rougeL_max = np.nanmax(rougeL_scores)

            metrics["rogue1"] = rouge1_max
            metrics["rogue2"] = rouge2_max
            metrics["rogueL"] = rougeL_max

        return metrics

    def bleu(self, refs, preds):
        """
        Returns `t5` style BLEU scores. See the related implementation:
        https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

        :param refs:
            A `list` of `list` of reference `str`s.
        :param preds:
            A `list` of predicted `str`s.
        """
        score = sacrebleu.corpus_bleu(
            preds,
            refs,
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            tokenize="intl",
            use_effective_order=False,
        ).score
        return score

    def rouge(self, refs, preds):
        """
        Returns `t5` style ROUGE scores. See the related implementation:
        https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

        :param refs:
            A `list` of reference `strs`.
        :param preds:
            A `list` of predicted `strs`.
        """
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types)
        # Add newlines between sentences to correctly compute `rougeLsum`.

        def _prepare_summary(summary):
            summary = summary.replace(" . ", ".\n")
            return summary

        # Accumulate confidence intervals.
        aggregator = scoring.BootstrapAggregator()
        for ref, pred in zip(refs, preds):
            ref = _prepare_summary(ref)
            pred = _prepare_summary(pred)
            aggregator.add_scores(scorer.score(ref, pred))
        result = aggregator.aggregate()
        return {type: result[type].mid.fmeasure * 100 for type in rouge_types}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        if self.MC:
            return {
                "acc_mc": mean,
                "acc_norm": mean,
            }
        else:
            return {
                "absolute_match_acc": mean,
                "bleurt": mean,
                "bleu": mean,
                "rogue1": mean,
                "rogue2": mean,
                "rogueL": mean,
            }

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        if self.MC:
            return {
                "acc_mc": True,
                "acc_norm": True,
            }
        else:
            return {
                "absolute_match": True,
                "bleurt": True,
                "bleu": True,
                "rogue1": True,
                "rogue2": True,
                "rogueL": True,
            }


# Only generative tasks
class AutoCategroization(BigBench_General):
    DATASET_NAME = "auto_categorization"


class CodeNames(BigBench_General):
    DATASET_NAME = "codenames"


class ObjectCounting(BigBench_General):
    DATASET_NAME: str = "object_counting"


# Only MC tasks
class Anachronisms(BigBench_General):
    DATASET_NAME = "anachronisms"


class AbstractNarrativeUnderstanding(BigBench_General):
    DATASET_NAME = "abstract_narrative_understanding"


class AnalogicalSimilarity(BigBench_General):
    DATASET_NAME = "analogical_similarity"


# Tasks with both kinds of tasks
class ArithmeticBB(BigBench_General):
    DATASET_NAME = "arithmetic"
