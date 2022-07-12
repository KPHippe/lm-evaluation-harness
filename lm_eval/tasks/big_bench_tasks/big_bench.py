# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import Task


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""

"""
Have to load each task individually, take a look at the 'dataset_name' fields

What happens when you try and load something from without the : 
```
ValueError: Config name is missing.
Please pick one among the available configs: ['abstract_narrative_understanding', 
'anachronisms', 'analogical_similarity', 'analytic_entailment', 'arithmetic', 
'ascii_word_recognition', 'authorship_verification', 'auto_categorization', 
'auto_debugging', 'bbq_lite_json', 'bridging_anaphora_resolution_barqa', 
'causal_judgment', 'cause_and_effect', 'checkmate_in_one', 'chess_state_tracking', 
'chinese_remainder_theorem', 'cifar10_classification', 'code_line_description', 
'codenames', 'color', 'common_morpheme', 'conceptual_combinations', 'conlang_translation',
 'contextual_parametric_knowledge_conflicts', 'crash_blossom', 'crass_ai', 
 'cryobiology_spanish', 'cryptonite', 'cs_algorithms', 'dark_humor_detection', 
 'date_understanding', 'disambiguation_qa', 'discourse_marker_prediction', 
 'disfl_qa', 'dyck_languages', 'elementary_math_qa', 'emoji_movie', 'emojis_emotion_prediction', 
 'empirical_judgments', 'english_proverbs', 'english_russian_proverbs', 'entailed_polarity', 
 'entailed_polarity_hindi', 'epistemic_reasoning', 'evaluating_information_essentiality', 
 'fact_checker', 'fantasy_reasoning', 'few_shot_nlg', 'figure_of_speech_detection', 
 'formal_fallacies_syllogisms_negation', 'gem', 'gender_inclusive_sentences_german', 
 'general_knowledge', 'geometric_shapes', 'goal_step_wikihow', 'gre_reading_comprehension', 
 'hhh_alignment', 'hindi_question_answering', 'hindu_knowledge', 'hinglish_toxicity', 
 'human_organs_senses', 'hyperbaton', 'identify_math_theorems', 'identify_odd_metaphor', 
 'implicatures', 'implicit_relations', 'intent_recognition', 
 'international_phonetic_alphabet_nli', 'international_phonetic_alphabet_transliterate', 
 'intersect_geometry', 'irony_identification', 'kanji_ascii', 'kannada', 'key_value_maps', 
 'known_unknowns', 'language_games', 'language_identification', 'linguistic_mappings', 
 'linguistics_puzzles', 'list_functions', 'logic_grid_puzzle', 'logical_args', 
 'logical_deduction', 'logical_fallacy_detection', 'logical_sequence', 
 'mathematical_induction', 'matrixshapes', 'metaphor_boolean', 'metaphor_understanding',
  'minute_mysteries_qa', 'misconceptions', 'misconceptions_russian', 'mnist_ascii', 
  'modified_arithmetic', 'moral_permissibility', 'movie_dialog_same_or_different', 
  'movie_recommendation', 'mult_data_wrangling', 'multiemo', 'natural_instructions', 
  'navigate', 'nonsense_words_grammar', 'novel_concepts', 'object_counting', 'odd_one_out', 
  'operators', 'paragraph_segmentation', 'parsinlu_qa', 'parsinlu_reading_comprehension', 
  'penguins_in_a_table', 'periodic_elements', 'persian_idioms', 'phrase_relatedness', 
  'physical_intuition', 'physics', 'physics_questions', 'play_dialog_same_or_different', 
  'polish_sequence_labeling', 'presuppositions_as_nli', 'qa_wikidata', 'question_selection', 
  'real_or_fake_text', 'reasoning_about_colored_objects', 'repeat_copy_logic', 'rephrase', 
  'riddle_sense', 'ruin_names', 'salient_translation_error_detection', 'scientific_press_release', 
  'semantic_parsing_in_context_sparc', 'semantic_parsing_spider', 'sentence_ambiguity', 
  'similarities_abstraction', 'simp_turing_concept', 'simple_arithmetic_json', 
  'simple_arithmetic_json_multiple_choice', 'simple_arithmetic_json_subtasks', 
  'simple_arithmetic_multiple_targets_json', 'simple_ethical_questions', 
  'simple_text_editing', 'snarks', 'social_iqa', 'social_support', 'sports_understanding', 
  'strange_stories', 'strategyqa', 'sufficient_information', 'suicide_risk', 
  'swahili_english_proverbs', 'swedish_to_german_proverbs', 'symbol_interpretation', 
  'temporal_sequences', 'tense', 'timedial', 'topical_chat', 'tracking_shuffled_objects', 
  'understanding_fables', 'undo_permutation', 'unit_conversion', 'unit_interpretation', 
  'unnatural_in_context_learning', 'vitaminc_fact_verification', 'what_is_the_tao', 
  'which_wiki_edit', 'winowhy', 'word_sorting', 'word_unscrambling']
```

"""

# TODO: Replace `NewTask` with the name of your Task.
class BigBench(Task):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "bigbench"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False

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
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]

    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        return doc

    def doc_to_text(self, doc):
        # TODO: Format the query prompt portion of the document example.
        return ""

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = ""
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
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        return []

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
        return {}

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
        return {}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {}
