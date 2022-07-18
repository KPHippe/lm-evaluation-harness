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

import tensorflow as tf

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
        try:
            # checking if both tasks are inherited
            if self.BOTH_TASKS:
                print(
                    "Both tasks present, defaulting to generative evaluation for now..."
                )
        except AttributeError:
            self.BOTH_TASKS = False

        if len(self.dataset["default"][0]["multiple_choice_targets"]) != 0:
            self.MC = True
        else:
            self.MC = False

        # Default metrics, change if needed
        self.metrics = ("absolute_match", "bleurt", "bleu")
        print(
            '"bleurt" in self.metrics and (not self.MC or self.BOTH_TASKS)',
            "bleurt" in self.metrics and (not self.MC or self.BOTH_TASKS),
        )
        if "bleurt" in self.metrics and (not self.MC or self.BOTH_TASKS):
            # Force bleurt to run on CPU, otherwise we will run out of memory on GPU
            # when running gpt-neox
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
        query = doc["inputs"]
        # for mc tasks, in gen tasks will be empty string
        choices = doc["multiple_choice_targets"]
        # for mc tasks, in gen tasks will be empty list
        try:
            gold = doc["multiple_choice_scores"].index(1)
        except ValueError:
            gold = -1
        # for generative tasks, present in MC tasks though
        targets = doc["targets"]
        return {
            "query": query,  # The query prompt.
            "choices": choices,  # The list of choices.
            "gold": gold,  # The integer used to index into the correct element of `"choices"`.
            "targets": targets,  # the answer(s) (for both gen and mc)
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
        if self.MC and not self.BOTH_TASKS:
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
        if self.MC and not self.BOTH_TASKS:
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
            with tf.device("cpu"):
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

        return {
            "acc_mc": mean,
            "acc_norm": mean,
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

        return {
            "acc_mc": True,
            "acc_norm": True,
            "absolute_match": True,
            "bleurt": True,
            "bleu": True,
            "rogue1": True,
            "rogue2": True,
            "rogueL": True,
        }


"""Original testing
"""
# # Only generative tasks
# class AutoCategroization(BigBench_General):
#     DATASET_NAME = "auto_categorization"


# class CodeNames(BigBench_General):
#     DATASET_NAME = "codenames"


# class ObjectCounting(BigBench_General):
#     DATASET_NAME: str = "object_counting"

# # Only MC tasks
# class Anachronisms(BigBench_General):
#     DATASET_NAME = "anachronisms"


# class AbstractNarrativeUnderstanding(BigBench_General):
#     DATASET_NAME = "abstract_narrative_understanding"


# class AnalogicalSimilarity(BigBench_General):
#     DATASET_NAME = "analogical_similarity"


# # Tasks with both kinds of tasks
# class Arithmetic(BigBench_General):
#     DATASET_NAME = "arithmetic"
#     BOTH_TASKS = True


class AbstractNarrativeUnderstanding(BigBench_General):
    DATASET_NAME = "abstract_narrative_understanding"


class AbstractionAndReasoningCorpus(BigBench_General):
    DATASET_NAME = "abstraction_and_reasoning_corpus"


class Anachronisms(BigBench_General):
    DATASET_NAME = "anachronisms"


class AnalogicalSimilarity(BigBench_General):
    DATASET_NAME = "analogical_similarity"


class AnalyticEntailment(BigBench_General):
    DATASET_NAME = "analytic_entailment"


class Arithmetic(BigBench_General):
    DATASET_NAME = "arithmetic"
    BOTH_TASKS = True


class AsciiWordRecognition(BigBench_General):
    DATASET_NAME = "ascii_word_recognition"


class AuthorshipVerification(BigBench_General):
    DATASET_NAME = "authorship_verification"


class AutoCategorization(BigBench_General):
    DATASET_NAME = "auto_categorization"


class AutoDebugging(BigBench_General):
    DATASET_NAME = "auto_debugging"


class BbqLite(BigBench_General):
    DATASET_NAME = "bbq_lite"


class BbqLiteJson(BigBench_General):
    DATASET_NAME = "bbq_lite_json"


class BiasFromProbabilities(BigBench_General):
    DATASET_NAME = "bias_from_probabilities"


class BooleanExpressions(BigBench_General):
    DATASET_NAME = "boolean_expressions"


class BridgingAnaphoraResolutionBarqa(BigBench_General):
    DATASET_NAME = "bridging_anaphora_resolution_barqa"


class Canary(BigBench_General):
    DATASET_NAME = "canary"


class CausalJudgment(BigBench_General):
    DATASET_NAME = "causal_judgment"


class CauseAndEffect(BigBench_General):
    DATASET_NAME = "cause_and_effect"


class CheckmateInOne(BigBench_General):
    DATASET_NAME = "checkmate_in_one"
    BOTH_TASKS = True


class ChessStateTracking(BigBench_General):
    DATASET_NAME = "chess_state_tracking"


class ChineseRemainderTheorem(BigBench_General):
    DATASET_NAME = "chinese_remainder_theorem"


class Cifar10Classification(BigBench_General):
    DATASET_NAME = "cifar10_classification"


class CodeLineDescription(BigBench_General):
    DATASET_NAME = "code_line_description"


class Codenames(BigBench_General):
    DATASET_NAME = "codenames"


class Color(BigBench_General):
    DATASET_NAME = "color"
    BOTH_TASKS = True


class Com2Sense(BigBench_General):
    DATASET_NAME = "com2sense"


class CommonMorpheme(BigBench_General):
    DATASET_NAME = "common_morpheme"


class ConceptualCombinations(BigBench_General):
    DATASET_NAME = "conceptual_combinations"


class ConlangTranslation(BigBench_General):
    DATASET_NAME = "conlang_translation"


class ContextDefinitionAlignment(BigBench_General):
    DATASET_NAME = "context_definition_alignment"


class ContextualParametricKnowledgeConflicts(BigBench_General):
    DATASET_NAME = "contextual_parametric_knowledge_conflicts"
    BOTH_TASKS = True


class Convinceme(BigBench_General):
    DATASET_NAME = "convinceme"


class CoqaConversationalQuestionAnswering(BigBench_General):
    DATASET_NAME = "coqa_conversational_question_answering"


class CrashBlossom(BigBench_General):
    DATASET_NAME = "crash_blossom"


class CrassAi(BigBench_General):
    DATASET_NAME = "crass_ai"


class CryobiologySpanish(BigBench_General):
    DATASET_NAME = "cryobiology_spanish"


class Cryptonite(BigBench_General):
    DATASET_NAME = "cryptonite"


class CsAlgorithms(BigBench_General):
    DATASET_NAME = "cs_algorithms"


class CycledLetters(BigBench_General):
    DATASET_NAME = "cycled_letters"


class DarkHumorDetection(BigBench_General):
    DATASET_NAME = "dark_humor_detection"


class DateUnderstanding(BigBench_General):
    DATASET_NAME = "date_understanding"


class DisambiguationQa(BigBench_General):
    DATASET_NAME = "disambiguation_qa"


class DiscourseMarkerPrediction(BigBench_General):
    DATASET_NAME = "discourse_marker_prediction"


class DisflQa(BigBench_General):
    DATASET_NAME = "disfl_qa"


class DiverseSocialBias(BigBench_General):
    DATASET_NAME = "diverse_social_bias"


class DyckLanguages(BigBench_General):
    DATASET_NAME = "dyck_languages"


class DynamicCounting(BigBench_General):
    DATASET_NAME = "dynamic_counting"


class ElementaryMathQa(BigBench_General):
    DATASET_NAME = "elementary_math_qa"


class EmojiMovie(BigBench_General):
    DATASET_NAME = "emoji_movie"
    BOTH_TASKS = True


class EmojisEmotionPrediction(BigBench_General):
    DATASET_NAME = "emojis_emotion_prediction"


class EmpiricalJudgments(BigBench_General):
    DATASET_NAME = "empirical_judgments"


class EnglishProverbs(BigBench_General):
    DATASET_NAME = "english_proverbs"


class EnglishRussianProverbs(BigBench_General):
    DATASET_NAME = "english_russian_proverbs"


class EntailedPolarity(BigBench_General):
    DATASET_NAME = "entailed_polarity"


class EntailedPolarityHindi(BigBench_General):
    DATASET_NAME = "entailed_polarity_hindi"


class EpistemicReasoning(BigBench_General):
    DATASET_NAME = "epistemic_reasoning"


class EvaluatingInformationEssentiality(BigBench_General):
    DATASET_NAME = "evaluating_information_essentiality"


class FactChecker(BigBench_General):
    DATASET_NAME = "fact_checker"


class FactualityOfSummary(BigBench_General):
    DATASET_NAME = "factuality_of_summary"


class FantasyReasoning(BigBench_General):
    DATASET_NAME = "fantasy_reasoning"


class FewShotNlg(BigBench_General):
    DATASET_NAME = "few_shot_nlg"


class FigureOfSpeechDetection(BigBench_General):
    DATASET_NAME = "figure_of_speech_detection"


class ForecastingSubquestions(BigBench_General):
    DATASET_NAME = "forecasting_subquestions"


class FormalFallaciesSyllogismsNegation(BigBench_General):
    DATASET_NAME = "formal_fallacies_syllogisms_negation"


class Gem(BigBench_General):
    DATASET_NAME = "gem"


class GenderInclusiveSentencesGerman(BigBench_General):
    DATASET_NAME = "gender_inclusive_sentences_german"


class GenderSensitivityChinese(BigBench_General):
    DATASET_NAME = "gender_sensitivity_chinese"


class GenderSensitivityEnglish(BigBench_General):
    DATASET_NAME = "gender_sensitivity_english"


class GeneralKnowledge(BigBench_General):
    DATASET_NAME = "general_knowledge"


class GeometricShapes(BigBench_General):
    DATASET_NAME = "geometric_shapes"
    BOTH_TASKS = True


class GoalStepWikihow(BigBench_General):
    DATASET_NAME = "goal_step_wikihow"


class GreReadingComprehension(BigBench_General):
    DATASET_NAME = "gre_reading_comprehension"


class HhhAlignment(BigBench_General):
    DATASET_NAME = "hhh_alignment"


class HighLowGame(BigBench_General):
    DATASET_NAME = "high_low_game"


class HindiQuestionAnswering(BigBench_General):
    DATASET_NAME = "hindi_question_answering"


class HinduKnowledge(BigBench_General):
    DATASET_NAME = "hindu_knowledge"


class HinglishToxicity(BigBench_General):
    DATASET_NAME = "hinglish_toxicity"


class HumanOrgansSenses(BigBench_General):
    DATASET_NAME = "human_organs_senses"


class Hyperbaton(BigBench_General):
    DATASET_NAME = "hyperbaton"


class IdentifyMathTheorems(BigBench_General):
    DATASET_NAME = "identify_math_theorems"


class IdentifyOddMetaphor(BigBench_General):
    DATASET_NAME = "identify_odd_metaphor"


class Implicatures(BigBench_General):
    DATASET_NAME = "implicatures"


class ImplicitRelations(BigBench_General):
    DATASET_NAME = "implicit_relations"


class IntentRecognition(BigBench_General):
    DATASET_NAME = "intent_recognition"


class InternationalPhoneticAlphabetNli(BigBench_General):
    DATASET_NAME = "international_phonetic_alphabet_nli"


class InternationalPhoneticAlphabetTransliterate(BigBench_General):
    DATASET_NAME = "international_phonetic_alphabet_transliterate"


class IntersectGeometry(BigBench_General):
    DATASET_NAME = "intersect_geometry"


class IronyIdentification(BigBench_General):
    DATASET_NAME = "irony_identification"


class KanjiAscii(BigBench_General):
    DATASET_NAME = "kanji_ascii"
    BOTH_TASKS = True


class Kannada(BigBench_General):
    DATASET_NAME = "kannada"


class KeyValueMaps(BigBench_General):
    DATASET_NAME = "key_value_maps"


class KnownUnknowns(BigBench_General):
    DATASET_NAME = "known_unknowns"


class LanguageGames(BigBench_General):
    DATASET_NAME = "language_games"


class LanguageIdentification(BigBench_General):
    DATASET_NAME = "language_identification"


class LinguisticMappings(BigBench_General):
    DATASET_NAME = "linguistic_mappings"


class LinguisticsPuzzles(BigBench_General):
    DATASET_NAME = "linguistics_puzzles"


class ListFunctions(BigBench_General):
    DATASET_NAME = "list_functions"


class LogicGridPuzzle(BigBench_General):
    DATASET_NAME = "logic_grid_puzzle"


class LogicalArgs(BigBench_General):
    DATASET_NAME = "logical_args"


class LogicalDeduction(BigBench_General):
    DATASET_NAME = "logical_deduction"


class LogicalFallacyDetection(BigBench_General):
    DATASET_NAME = "logical_fallacy_detection"


class LogicalSequence(BigBench_General):
    DATASET_NAME = "logical_sequence"


class LongContextIntegration(BigBench_General):
    DATASET_NAME = "long_context_integration"


class MathematicalInduction(BigBench_General):
    DATASET_NAME = "mathematical_induction"


class Matrixshapes(BigBench_General):
    DATASET_NAME = "matrixshapes"


class MedicalQuestionsRussian(BigBench_General):
    DATASET_NAME = "medical_questions_russian"


class MetaphorBoolean(BigBench_General):
    DATASET_NAME = "metaphor_boolean"


class MetaphorUnderstanding(BigBench_General):
    DATASET_NAME = "metaphor_understanding"


class MinuteMysteriesQa(BigBench_General):
    DATASET_NAME = "minute_mysteries_qa"
    BOTH_TASKS = True


class Misconceptions(BigBench_General):
    DATASET_NAME = "misconceptions"


class MisconceptionsRussian(BigBench_General):
    DATASET_NAME = "misconceptions_russian"


class MnistAscii(BigBench_General):
    DATASET_NAME = "mnist_ascii"


class ModifiedArithmetic(BigBench_General):
    DATASET_NAME = "modified_arithmetic"


class MoralPermissibility(BigBench_General):
    DATASET_NAME = "moral_permissibility"


class MovieDialogSameOrDifferent(BigBench_General):
    DATASET_NAME = "movie_dialog_same_or_different"


class MovieRecommendation(BigBench_General):
    DATASET_NAME = "movie_recommendation"


class MultDataWrangling(BigBench_General):
    DATASET_NAME = "mult_data_wrangling"


class Multiemo(BigBench_General):
    DATASET_NAME = "multiemo"


class MultistepArithmetic(BigBench_General):
    DATASET_NAME = "multistep_arithmetic"


class MuslimViolenceBias(BigBench_General):
    DATASET_NAME = "muslim_violence_bias"


class NaturalInstructions(BigBench_General):
    DATASET_NAME = "natural_instructions"


class Navigate(BigBench_General):
    DATASET_NAME = "navigate"


class NonsenseWordsGrammar(BigBench_General):
    DATASET_NAME = "nonsense_words_grammar"


class NovelConcepts(BigBench_General):
    DATASET_NAME = "novel_concepts"


class ObjectCounting(BigBench_General):
    DATASET_NAME = "object_counting"


class OddOneOut(BigBench_General):
    DATASET_NAME = "odd_one_out"


class Operators(BigBench_General):
    DATASET_NAME = "operators"


class ParagraphSegmentation(BigBench_General):
    DATASET_NAME = "paragraph_segmentation"


class ParsinluQa(BigBench_General):
    DATASET_NAME = "parsinlu_qa"


class ParsinluReadingComprehension(BigBench_General):
    DATASET_NAME = "parsinlu_reading_comprehension"


class PenguinsInATable(BigBench_General):
    DATASET_NAME = "penguins_in_a_table"
    BOTH_TASKS = True


class PeriodicElements(BigBench_General):
    DATASET_NAME = "periodic_elements"
    BOTH_TASKS = True


class PersianIdioms(BigBench_General):
    DATASET_NAME = "persian_idioms"


class PhraseRelatedness(BigBench_General):
    DATASET_NAME = "phrase_relatedness"


class PhysicalIntuition(BigBench_General):
    DATASET_NAME = "physical_intuition"


class Physics(BigBench_General):
    DATASET_NAME = "physics"


class PhysicsQuestions(BigBench_General):
    DATASET_NAME = "physics_questions"


class PlayDialogSameOrDifferent(BigBench_General):
    DATASET_NAME = "play_dialog_same_or_different"


class PolishSequenceLabeling(BigBench_General):
    DATASET_NAME = "polish_sequence_labeling"


class PresuppositionsAsNli(BigBench_General):
    DATASET_NAME = "presuppositions_as_nli"


class ProgramSynthesis(BigBench_General):
    DATASET_NAME = "program_synthesis"


class ProteinInteractingSites(BigBench_General):
    DATASET_NAME = "protein_interacting_sites"


class PythonProgrammingChallenge(BigBench_General):
    DATASET_NAME = "python_programming_challenge"


class QaWikidata(BigBench_General):
    DATASET_NAME = "qa_wikidata"


class QuestionAnswerCreation(BigBench_General):
    DATASET_NAME = "question_answer_creation"


class QuestionSelection(BigBench_General):
    DATASET_NAME = "question_selection"


class RealOrFakeText(BigBench_General):
    DATASET_NAME = "real_or_fake_text"


class ReasoningAboutColoredObjects(BigBench_General):
    DATASET_NAME = "reasoning_about_colored_objects"


class RepeatCopyLogic(BigBench_General):
    DATASET_NAME = "repeat_copy_logic"


class Rephrase(BigBench_General):
    DATASET_NAME = "rephrase"


class RiddleSense(BigBench_General):
    DATASET_NAME = "riddle_sense"


class RootsOptimizationAndGames(BigBench_General):
    DATASET_NAME = "roots_optimization_and_games"


class RuinNames(BigBench_General):
    DATASET_NAME = "ruin_names"


class SalientTranslationErrorDetection(BigBench_General):
    DATASET_NAME = "salient_translation_error_detection"


class ScientificPressRelease(BigBench_General):
    DATASET_NAME = "scientific_press_release"


class SelfAwareness(BigBench_General):
    DATASET_NAME = "self_awareness"


class SelfEvaluationCourtroom(BigBench_General):
    DATASET_NAME = "self_evaluation_courtroom"


class SelfEvaluationTutoring(BigBench_General):
    DATASET_NAME = "self_evaluation_tutoring"


class SemanticParsingInContextSparc(BigBench_General):
    DATASET_NAME = "semantic_parsing_in_context_sparc"


class SemanticParsingSpider(BigBench_General):
    DATASET_NAME = "semantic_parsing_spider"


class SentenceAmbiguity(BigBench_General):
    DATASET_NAME = "sentence_ambiguity"


class SimilaritiesAbstraction(BigBench_General):
    DATASET_NAME = "similarities_abstraction"
    BOTH_TASKS = True


class SimpTuringConcept(BigBench_General):
    DATASET_NAME = "simp_turing_concept"


class SimpleArithmetic(BigBench_General):
    DATASET_NAME = "simple_arithmetic"


class SimpleArithmeticJson(BigBench_General):
    DATASET_NAME = "simple_arithmetic_json"


class SimpleArithmeticJsonMultipleChoice(BigBench_General):
    DATASET_NAME = "simple_arithmetic_json_multiple_choice"


class SimpleArithmeticJsonSubtasks(BigBench_General):
    DATASET_NAME = "simple_arithmetic_json_subtasks"


class SimpleArithmeticMultipleTargetsJson(BigBench_General):
    DATASET_NAME = "simple_arithmetic_multiple_targets_json"


class SimpleEthicalQuestions(BigBench_General):
    DATASET_NAME = "simple_ethical_questions"


class SimpleTextEditing(BigBench_General):
    DATASET_NAME = "simple_text_editing"


class Snarks(BigBench_General):
    DATASET_NAME = "snarks"


class SocialIqa(BigBench_General):
    DATASET_NAME = "social_iqa"


class SocialSupport(BigBench_General):
    DATASET_NAME = "social_support"


class SpellingBee(BigBench_General):
    DATASET_NAME = "spelling_bee"


class SportsUnderstanding(BigBench_General):
    DATASET_NAME = "sports_understanding"


class SquadShifts(BigBench_General):
    DATASET_NAME = "squad_shifts"


class StrangeStories(BigBench_General):
    DATASET_NAME = "strange_stories"


class Strategyqa(BigBench_General):
    DATASET_NAME = "strategyqa"
    BOTH_TASKS = True


class SubjectVerbAgreement(BigBench_General):
    DATASET_NAME = "subject_verb_agreement"


class Sudoku(BigBench_General):
    DATASET_NAME = "sudoku"


class SufficientInformation(BigBench_General):
    DATASET_NAME = "sufficient_information"


class SuicideRisk(BigBench_General):
    DATASET_NAME = "suicide_risk"


class SwahiliEnglishProverbs(BigBench_General):
    DATASET_NAME = "swahili_english_proverbs"


class SwedishToGermanProverbs(BigBench_General):
    DATASET_NAME = "swedish_to_german_proverbs"


class SymbolInterpretation(BigBench_General):
    DATASET_NAME = "symbol_interpretation"


class Taboo(BigBench_General):
    DATASET_NAME = "taboo"


class Talkdown(BigBench_General):
    DATASET_NAME = "talkdown"


class TemporalSequences(BigBench_General):
    DATASET_NAME = "temporal_sequences"


class Tense(BigBench_General):
    DATASET_NAME = "tense"


class TextNavigationGame(BigBench_General):
    DATASET_NAME = "text_navigation_game"


class Timedial(BigBench_General):
    DATASET_NAME = "timedial"


class TopicalChat(BigBench_General):
    DATASET_NAME = "topical_chat"


class TrackingShuffledObjects(BigBench_General):
    DATASET_NAME = "tracking_shuffled_objects"


class TrainingOnTestSet(BigBench_General):
    DATASET_NAME = "training_on_test_set"


class TruthfulQa(BigBench_General):
    DATASET_NAME = "truthful_qa"


class TwentyQuestions(BigBench_General):
    DATASET_NAME = "twenty_questions"


class UnderstandingFables(BigBench_General):
    DATASET_NAME = "understanding_fables"


class UndoPermutation(BigBench_General):
    DATASET_NAME = "undo_permutation"


class UnitConversion(BigBench_General):
    DATASET_NAME = "unit_conversion"
    BOTH_TASKS = True


class UnitInterpretation(BigBench_General):
    DATASET_NAME = "unit_interpretation"


class UnnaturalInContextLearning(BigBench_General):
    DATASET_NAME = "unnatural_in_context_learning"


class Unqover(BigBench_General):
    DATASET_NAME = "unqover"


class VitamincFactVerification(BigBench_General):
    DATASET_NAME = "vitaminc_fact_verification"


class WebOfLies(BigBench_General):
    DATASET_NAME = "web_of_lies"


class WhatIsTheTao(BigBench_General):
    DATASET_NAME = "what_is_the_tao"


class WhichWikiEdit(BigBench_General):
    DATASET_NAME = "which_wiki_edit"


class Winowhy(BigBench_General):
    DATASET_NAME = "winowhy"


class WordProblemsOnSetsAndGraphs(BigBench_General):
    DATASET_NAME = "word_problems_on_sets_and_graphs"


class WordSorting(BigBench_General):
    DATASET_NAME = "word_sorting"


class WordUnscrambling(BigBench_General):
    DATASET_NAME = "word_unscrambling"


class YesNoBlackWhite(BigBench_General):
    DATASET_NAME = "yes_no_black_white"
