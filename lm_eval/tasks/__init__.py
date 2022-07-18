from pprint import pprint
from typing import List, Union

import sacrebleu
import lm_eval.base

from .big_bench_tasks import big_bench_all as bb

from . import radbio

from . import superglue
from . import glue
from . import arc
from . import coqa
from . import race
from . import webqs
from . import anli
from . import wsc273
from . import winogrande
from . import quac
from . import hellaswag
from . import swag
from . import openbookqa
from . import squad
from . import naturalqs
from . import sat
from . import arithmetic
from . import lambada
from . import piqa
from . import prost
from . import mc_taco
from . import triviaqa
from . import pubmedqa
from . import sciq
from . import qasper
from . import qa4mre
from . import translation
from . import headqa
from . import mathqa
from . import hendrycks_ethics
from . import drop
from . import unscramble
from . import logiqa
from . import hendrycks_test
from . import hendrycks_math
from . import cbt
from . import lambada_cloze
from . import pile
from . import wikitext
from . import lambada_multilingual
from . import mutual
from . import truthfulqa
from . import blimp
from . import asdiv
from . import gsm8k
from . import storycloze

########################################
# Translation tasks
########################################

# 6 total
gpt3_translation_benchmarks = {
    "wmt14": ["en-fr", "fr-en"],  # French
    "wmt16": ["en-ro", "ro-en", "de-en", "en-de"],  # German, Romanian
}


# 28 total
selected_translation_benchmarks = {
    **gpt3_translation_benchmarks,
    "wmt20": sacrebleu.get_langpairs_for_testset("wmt20"),
    "iwslt17": ["en-ar", "ar-en"],  # Arabic
}

# 319 total
all_translation_benchmarks = {
    ts: sacrebleu.get_langpairs_for_testset(ts)
    for ts in sacrebleu.get_available_testsets()
}


########################################
# All tasks
########################################


TASK_REGISTRY = {
    # radbio
    "isInSystemQA": radbio.isInSystemQA,
    "goAHumanQA": radbio.goAHumanQA,
    "goARadiationResponseQA": radbio.goARadiationResponseQA,
    "ppiHumanQA": radbio.ppiHumanQA,
    "humanPathwaysQA": radbio.humanPathwaysQA,
    # BIG-Bench tasks
    # "anachronisms": bb.Anachronisms,
    # "abstract_narrative_understanding": bb.AbstractNarrativeUnderstanding,
    # "analogical_similarity": bb.AnalogicalSimilarity,
    # "auto_categorization": bb.AutoCategroization,
    # "arithmetic": bb.Arithmetic,
    # "codenames": bb.CodeNames,
    # "object_counting": bb.ObjectCounting,
    "abstract_narrative_understanding": bb.AbstractNarrativeUnderstanding,
    "abstraction_and_reasoning_corpus": bb.AbstractionAndReasoningCorpus,
    "anachronisms": bb.Anachronisms,
    "analogical_similarity": bb.AnalogicalSimilarity,
    "analytic_entailment": bb.AnalyticEntailment,
    "arithmetic": bb.Arithmetic,
    "ascii_word_recognition": bb.AsciiWordRecognition,
    "authorship_verification": bb.AuthorshipVerification,
    "auto_categorization": bb.AutoCategorization,
    "auto_debugging": bb.AutoDebugging,
    "bbq_lite": bb.BbqLite,
    "bbq_lite_json": bb.BbqLiteJson,
    "bias_from_probabilities": bb.BiasFromProbabilities,
    "boolean_expressions": bb.BooleanExpressions,
    "bridging_anaphora_resolution_barqa": bb.BridgingAnaphoraResolutionBarqa,
    "canary": bb.Canary,
    "causal_judgment": bb.CausalJudgment,
    "cause_and_effect": bb.CauseAndEffect,
    "checkmate_in_one": bb.CheckmateInOne,
    "chess_state_tracking": bb.ChessStateTracking,
    "chinese_remainder_theorem": bb.ChineseRemainderTheorem,
    "cifar10_classification": bb.Cifar10Classification,
    "code_line_description": bb.CodeLineDescription,
    "codenames": bb.Codenames,
    "color": bb.Color,
    "com2sense": bb.Com2Sense,
    "common_morpheme": bb.CommonMorpheme,
    "conceptual_combinations": bb.ConceptualCombinations,
    "conlang_translation": bb.ConlangTranslation,
    "context_definition_alignment": bb.ContextDefinitionAlignment,
    "contextual_parametric_knowledge_conflicts": bb.ContextualParametricKnowledgeConflicts,
    "convinceme": bb.Convinceme,
    "coqa_conversational_question_answering": bb.CoqaConversationalQuestionAnswering,
    "crash_blossom": bb.CrashBlossom,
    "crass_ai": bb.CrassAi,
    "cryobiology_spanish": bb.CryobiologySpanish,
    "cryptonite": bb.Cryptonite,
    "cs_algorithms": bb.CsAlgorithms,
    "cycled_letters": bb.CycledLetters,
    "dark_humor_detection": bb.DarkHumorDetection,
    "date_understanding": bb.DateUnderstanding,
    "disambiguation_qa": bb.DisambiguationQa,
    "discourse_marker_prediction": bb.DiscourseMarkerPrediction,
    "disfl_qa": bb.DisflQa,
    "diverse_social_bias": bb.DiverseSocialBias,
    "dyck_languages": bb.DyckLanguages,
    "dynamic_counting": bb.DynamicCounting,
    "elementary_math_qa": bb.ElementaryMathQa,
    "emoji_movie": bb.EmojiMovie,
    "emojis_emotion_prediction": bb.EmojisEmotionPrediction,
    "empirical_judgments": bb.EmpiricalJudgments,
    "english_proverbs": bb.EnglishProverbs,
    "english_russian_proverbs": bb.EnglishRussianProverbs,
    "entailed_polarity": bb.EntailedPolarity,
    "entailed_polarity_hindi": bb.EntailedPolarityHindi,
    "epistemic_reasoning": bb.EpistemicReasoning,
    "evaluating_information_essentiality": bb.EvaluatingInformationEssentiality,
    "fact_checker": bb.FactChecker,
    "factuality_of_summary": bb.FactualityOfSummary,
    "fantasy_reasoning": bb.FantasyReasoning,
    "few_shot_nlg": bb.FewShotNlg,
    "figure_of_speech_detection": bb.FigureOfSpeechDetection,
    "forecasting_subquestions": bb.ForecastingSubquestions,
    "formal_fallacies_syllogisms_negation": bb.FormalFallaciesSyllogismsNegation,
    "gem": bb.Gem,
    "gender_inclusive_sentences_german": bb.GenderInclusiveSentencesGerman,
    "gender_sensitivity_chinese": bb.GenderSensitivityChinese,
    "gender_sensitivity_english": bb.GenderSensitivityEnglish,
    "general_knowledge": bb.GeneralKnowledge,
    "geometric_shapes": bb.GeometricShapes,
    "goal_step_wikihow": bb.GoalStepWikihow,
    "gre_reading_comprehension": bb.GreReadingComprehension,
    "hhh_alignment": bb.HhhAlignment,
    "high_low_game": bb.HighLowGame,
    "hindi_question_answering": bb.HindiQuestionAnswering,
    "hindu_knowledge": bb.HinduKnowledge,
    "hinglish_toxicity": bb.HinglishToxicity,
    "human_organs_senses": bb.HumanOrgansSenses,
    "hyperbaton": bb.Hyperbaton,
    "identify_math_theorems": bb.IdentifyMathTheorems,
    "identify_odd_metaphor": bb.IdentifyOddMetaphor,
    "implicatures": bb.Implicatures,
    "implicit_relations": bb.ImplicitRelations,
    "intent_recognition": bb.IntentRecognition,
    "international_phonetic_alphabet_nli": bb.InternationalPhoneticAlphabetNli,
    "international_phonetic_alphabet_transliterate": bb.InternationalPhoneticAlphabetTransliterate,
    "intersect_geometry": bb.IntersectGeometry,
    "irony_identification": bb.IronyIdentification,
    "kanji_ascii": bb.KanjiAscii,
    "kannada": bb.Kannada,
    "key_value_maps": bb.KeyValueMaps,
    "known_unknowns": bb.KnownUnknowns,
    "language_games": bb.LanguageGames,
    "language_identification": bb.LanguageIdentification,
    "linguistic_mappings": bb.LinguisticMappings,
    "linguistics_puzzles": bb.LinguisticsPuzzles,
    "list_functions": bb.ListFunctions,
    "logic_grid_puzzle": bb.LogicGridPuzzle,
    "logical_args": bb.LogicalArgs,
    "logical_deduction": bb.LogicalDeduction,
    "logical_fallacy_detection": bb.LogicalFallacyDetection,
    "logical_sequence": bb.LogicalSequence,
    "long_context_integration": bb.LongContextIntegration,
    "mathematical_induction": bb.MathematicalInduction,
    "matrixshapes": bb.Matrixshapes,
    "medical_questions_russian": bb.MedicalQuestionsRussian,
    "metaphor_boolean": bb.MetaphorBoolean,
    "metaphor_understanding": bb.MetaphorUnderstanding,
    "minute_mysteries_qa": bb.MinuteMysteriesQa,
    "misconceptions": bb.Misconceptions,
    "misconceptions_russian": bb.MisconceptionsRussian,
    "mnist_ascii": bb.MnistAscii,
    "modified_arithmetic": bb.ModifiedArithmetic,
    "moral_permissibility": bb.MoralPermissibility,
    "movie_dialog_same_or_different": bb.MovieDialogSameOrDifferent,
    "movie_recommendation": bb.MovieRecommendation,
    "mult_data_wrangling": bb.MultDataWrangling,
    "multiemo": bb.Multiemo,
    "multistep_arithmetic": bb.MultistepArithmetic,
    "muslim_violence_bias": bb.MuslimViolenceBias,
    "natural_instructions": bb.NaturalInstructions,
    "navigate": bb.Navigate,
    "nonsense_words_grammar": bb.NonsenseWordsGrammar,
    "novel_concepts": bb.NovelConcepts,
    "object_counting": bb.ObjectCounting,
    "odd_one_out": bb.OddOneOut,
    "operators": bb.Operators,
    "paragraph_segmentation": bb.ParagraphSegmentation,
    "parsinlu_qa": bb.ParsinluQa,
    "parsinlu_reading_comprehension": bb.ParsinluReadingComprehension,
    "penguins_in_a_table": bb.PenguinsInATable,
    "periodic_elements": bb.PeriodicElements,
    "persian_idioms": bb.PersianIdioms,
    "phrase_relatedness": bb.PhraseRelatedness,
    "physical_intuition": bb.PhysicalIntuition,
    "physics": bb.Physics,
    "physics_questions": bb.PhysicsQuestions,
    "play_dialog_same_or_different": bb.PlayDialogSameOrDifferent,
    "polish_sequence_labeling": bb.PolishSequenceLabeling,
    "presuppositions_as_nli": bb.PresuppositionsAsNli,
    "program_synthesis": bb.ProgramSynthesis,
    "protein_interacting_sites": bb.ProteinInteractingSites,
    "python_programming_challenge": bb.PythonProgrammingChallenge,
    "qa_wikidata": bb.QaWikidata,
    "question_answer_creation": bb.QuestionAnswerCreation,
    "question_selection": bb.QuestionSelection,
    "real_or_fake_text": bb.RealOrFakeText,
    "reasoning_about_colored_objects": bb.ReasoningAboutColoredObjects,
    "repeat_copy_logic": bb.RepeatCopyLogic,
    "rephrase": bb.Rephrase,
    "riddle_sense": bb.RiddleSense,
    "roots_optimization_and_games": bb.RootsOptimizationAndGames,
    "ruin_names": bb.RuinNames,
    "salient_translation_error_detection": bb.SalientTranslationErrorDetection,
    "scientific_press_release": bb.ScientificPressRelease,
    "self_awareness": bb.SelfAwareness,
    "self_evaluation_courtroom": bb.SelfEvaluationCourtroom,
    "self_evaluation_tutoring": bb.SelfEvaluationTutoring,
    "semantic_parsing_in_context_sparc": bb.SemanticParsingInContextSparc,
    "semantic_parsing_spider": bb.SemanticParsingSpider,
    "sentence_ambiguity": bb.SentenceAmbiguity,
    "similarities_abstraction": bb.SimilaritiesAbstraction,
    "simp_turing_concept": bb.SimpTuringConcept,
    "simple_arithmetic": bb.SimpleArithmetic,
    "simple_arithmetic_json": bb.SimpleArithmeticJson,
    "simple_arithmetic_json_multiple_choice": bb.SimpleArithmeticJsonMultipleChoice,
    "simple_arithmetic_json_subtasks": bb.SimpleArithmeticJsonSubtasks,
    "simple_arithmetic_multiple_targets_json": bb.SimpleArithmeticMultipleTargetsJson,
    "simple_ethical_questions": bb.SimpleEthicalQuestions,
    "simple_text_editing": bb.SimpleTextEditing,
    "snarks": bb.Snarks,
    "social_iqa": bb.SocialIqa,
    "social_support": bb.SocialSupport,
    "spelling_bee": bb.SpellingBee,
    "sports_understanding": bb.SportsUnderstanding,
    "squad_shifts": bb.SquadShifts,
    "strange_stories": bb.StrangeStories,
    "strategyqa": bb.Strategyqa,
    "subject_verb_agreement": bb.SubjectVerbAgreement,
    "sudoku": bb.Sudoku,
    "sufficient_information": bb.SufficientInformation,
    "suicide_risk": bb.SuicideRisk,
    "swahili_english_proverbs": bb.SwahiliEnglishProverbs,
    "swedish_to_german_proverbs": bb.SwedishToGermanProverbs,
    "symbol_interpretation": bb.SymbolInterpretation,
    "taboo": bb.Taboo,
    "talkdown": bb.Talkdown,
    "temporal_sequences": bb.TemporalSequences,
    "tense": bb.Tense,
    "text_navigation_game": bb.TextNavigationGame,
    "timedial": bb.Timedial,
    "topical_chat": bb.TopicalChat,
    "tracking_shuffled_objects": bb.TrackingShuffledObjects,
    "training_on_test_set": bb.TrainingOnTestSet,
    "truthful_qa": bb.TruthfulQa,
    "twenty_questions": bb.TwentyQuestions,
    "understanding_fables": bb.UnderstandingFables,
    "undo_permutation": bb.UndoPermutation,
    "unit_conversion": bb.UnitConversion,
    "unit_interpretation": bb.UnitInterpretation,
    "unnatural_in_context_learning": bb.UnnaturalInContextLearning,
    "unqover": bb.Unqover,
    "vitaminc_fact_verification": bb.VitamincFactVerification,
    "web_of_lies": bb.WebOfLies,
    "what_is_the_tao": bb.WhatIsTheTao,
    "which_wiki_edit": bb.WhichWikiEdit,
    "winowhy": bb.Winowhy,
    "word_problems_on_sets_and_graphs": bb.WordProblemsOnSetsAndGraphs,
    "word_sorting": bb.WordSorting,
    "word_unscrambling": bb.WordUnscrambling,
    "yes_no_black_white": bb.YesNoBlackWhite,
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mnli_mismatched": glue.MNLIMismatched,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    # "stsb": glue.STSB, # not implemented yet
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "record": superglue.ReCoRD,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    # Order by benchmark/genre?
    "coqa": coqa.CoQA,
    "drop": drop.DROP,
    "lambada": lambada.LAMBADA,
    "lambada_cloze": lambada_cloze.LAMBADA_cloze,
    # multilingual lambada
    **lambada_multilingual.construct_tasks(),
    "wikitext": wikitext.WikiText,
    # "cbt-cn": cbt.CBTCN, # disabled pending context length fix
    # "cbt-ne": cbt.CBTNE, # disabled pending context length fix
    "piqa": piqa.PiQA,
    "prost": prost.PROST,
    "mc_taco": mc_taco.MCTACO,
    # Science related
    "pubmedqa": pubmedqa.Pubmed_QA,
    "sciq": sciq.SciQ,
    "qasper": qasper.QASPER,
    "qa4mre_2011": qa4mre.QA4MRE_2011,
    "qa4mre_2012": qa4mre.QA4MRE_2012,
    "qa4mre_2013": qa4mre.QA4MRE_2013,
    "triviaqa": triviaqa.TriviaQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    # "quac": quac.QuAC, # not implemented yet
    "logiqa": logiqa.LogiQA,
    "hellaswag": hellaswag.HellaSwag,
    "swag": swag.SWAG,
    "openbookqa": openbookqa.OpenBookQA,
    "squad2": squad.SQuAD2,
    "race": race.RACE,
    # "naturalqs": naturalqs.NaturalQs, # not implemented yet
    "headqa": headqa.HeadQAEsDeprecated,  # for backwards compat - headqa used to default to es
    "headqa_es": headqa.HeadQAEs,
    "headqa_en": headqa.HeadQAEn,
    "mathqa": mathqa.MathQA,
    "webqs": webqs.WebQs,
    "wsc273": wsc273.WinogradSchemaChallenge273,
    "winogrande": winogrande.Winogrande,
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "ethics_cm": hendrycks_ethics.EthicsCM,
    "ethics_deontology": hendrycks_ethics.EthicsDeontology,
    "ethics_justice": hendrycks_ethics.EthicsJustice,
    "ethics_utilitarianism_original": hendrycks_ethics.EthicsUtilitarianismOriginal,
    "ethics_utilitarianism": hendrycks_ethics.EthicsUtilitarianism,
    "ethics_virtue": hendrycks_ethics.EthicsVirtue,
    "truthfulqa_mc": truthfulqa.TruthfulQAMultipleChoice,
    "truthfulqa_gen": truthfulqa.TruthfulQAGeneration,
    # dialogue
    "mutual": mutual.MuTual,
    "mutual_plus": mutual.MuTualPlus,
    # math
    "math_algebra": hendrycks_math.MathAlgebra,
    "math_counting_and_prob": hendrycks_math.MathCountingAndProbability,
    "math_geometry": hendrycks_math.MathGeometry,
    "math_intermediate_algebra": hendrycks_math.MathIntermediateAlgebra,
    "math_num_theory": hendrycks_math.MathNumberTheory,
    "math_prealgebra": hendrycks_math.MathPrealgebra,
    "math_precalc": hendrycks_math.MathPrecalculus,
    "math_asdiv": asdiv.Asdiv,
    "gsm8k": gsm8k.GradeSchoolMath8K,
    # arithmetic
    "arithmetic_2da": arithmetic.Arithmetic2DPlus,
    "arithmetic_2ds": arithmetic.Arithmetic2DMinus,
    "arithmetic_3da": arithmetic.Arithmetic3DPlus,
    "arithmetic_3ds": arithmetic.Arithmetic3DMinus,
    "arithmetic_4da": arithmetic.Arithmetic4DPlus,
    "arithmetic_4ds": arithmetic.Arithmetic4DMinus,
    "arithmetic_5da": arithmetic.Arithmetic5DPlus,
    "arithmetic_5ds": arithmetic.Arithmetic5DMinus,
    "arithmetic_2dm": arithmetic.Arithmetic2DMultiplication,
    "arithmetic_1dc": arithmetic.Arithmetic1DComposite,
    # TODO Perhaps make these groups of tasks
    #   e.g. anli, arithmetic, openai_translations, harness_translations
    # hendrycksTest (57 tasks)
    **hendrycks_test.create_all_tasks(),
    # e.g. wmt14-fr-en
    **translation.create_tasks_from_benchmarks(gpt3_translation_benchmarks),
    # chef's selection, mostly wmt20
    **translation.create_tasks_from_benchmarks(selected_translation_benchmarks),
    # Word Scrambling and Manipulation Tasks
    "anagrams1": unscramble.Anagrams1,
    "anagrams2": unscramble.Anagrams2,
    "cycle_letters": unscramble.CycleLetters,
    "random_insertion": unscramble.RandomInsertion,
    "reversed_words": unscramble.ReversedWords,
    # Pile
    "pile_arxiv": pile.PileArxiv,
    "pile_books3": pile.PileBooks3,
    "pile_bookcorpus2": pile.PileBookCorpus2,
    "pile_dm-mathematics": pile.PileDmMathematics,
    "pile_enron": pile.PileEnron,
    "pile_europarl": pile.PileEuroparl,
    "pile_freelaw": pile.PileFreeLaw,
    "pile_github": pile.PileGithub,
    "pile_gutenberg": pile.PileGutenberg,
    "pile_hackernews": pile.PileHackernews,
    "pile_nih-exporter": pile.PileNIHExporter,
    "pile_opensubtitles": pile.PileOpenSubtitles,
    "pile_openwebtext2": pile.PileOpenWebText2,
    "pile_philpapers": pile.PilePhilPapers,
    "pile_pile-cc": pile.PilePileCc,
    "pile_pubmed-abstracts": pile.PilePubmedAbstracts,
    "pile_pubmed-central": pile.PilePubmedCentral,
    "pile_stackexchange": pile.PileStackExchange,
    "pile_uspto": pile.PileUspto,
    "pile_ubuntu-irc": pile.PileUbuntuIrc,
    "pile_wikipedia": pile.PileWikipedia,
    "pile_youtubesubtitles": pile.PileYoutubeSubtitles,
    # BLiMP
    "blimp_adjunct_island": blimp.BlimpAdjunctIsland,
    "blimp_anaphor_gender_agreement": blimp.BlimpAnaphorGenderAgreement,
    "blimp_anaphor_number_agreement": blimp.BlimpAnaphorNumberAgreement,
    "blimp_animate_subject_passive": blimp.BlimpAnimateSubjectPassive,
    "blimp_animate_subject_trans": blimp.BlimpAnimateSubjectTrans,
    "blimp_causative": blimp.BlimpCausative,
    "blimp_complex_NP_island": blimp.BlimpComplex_NPIsland,
    "blimp_coordinate_structure_constraint_complex_left_branch": blimp.BlimpCoordinateStructureConstraintComplexLeftBranch,
    "blimp_coordinate_structure_constraint_object_extraction": blimp.BlimpCoordinateStructureConstraintObjectExtraction,
    "blimp_determiner_noun_agreement_1": blimp.BlimpDeterminerNounAgreement_1,
    "blimp_determiner_noun_agreement_2": blimp.BlimpDeterminerNounAgreement_2,
    "blimp_determiner_noun_agreement_irregular_1": blimp.BlimpDeterminerNounAgreementIrregular_1,
    "blimp_determiner_noun_agreement_irregular_2": blimp.BlimpDeterminerNounAgreementIrregular_2,
    "blimp_determiner_noun_agreement_with_adj_2": blimp.BlimpDeterminerNounAgreementWithAdj_2,
    "blimp_determiner_noun_agreement_with_adj_irregular_1": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_1,
    "blimp_determiner_noun_agreement_with_adj_irregular_2": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_2,
    "blimp_determiner_noun_agreement_with_adjective_1": blimp.BlimpDeterminerNounAgreementWithAdjective_1,
    "blimp_distractor_agreement_relational_noun": blimp.BlimpDistractorAgreementRelationalNoun,
    "blimp_distractor_agreement_relative_clause": blimp.BlimpDistractorAgreementRelativeClause,
    "blimp_drop_argument": blimp.BlimpDropArgument,
    "blimp_ellipsis_n_bar_1": blimp.BlimpEllipsisNBar_1,
    "blimp_ellipsis_n_bar_2": blimp.BlimpEllipsisNBar_2,
    "blimp_existential_there_object_raising": blimp.BlimpExistentialThereObjectRaising,
    "blimp_existential_there_quantifiers_1": blimp.BlimpExistentialThereQuantifiers_1,
    "blimp_existential_there_quantifiers_2": blimp.BlimpExistentialThereQuantifiers_2,
    "blimp_existential_there_subject_raising": blimp.BlimpExistentialThereSubjectRaising,
    "blimp_expletive_it_object_raising": blimp.BlimpExpletiveItObjectRaising,
    "blimp_inchoative": blimp.BlimpInchoative,
    "blimp_intransitive": blimp.BlimpIntransitive,
    "blimp_irregular_past_participle_adjectives": blimp.BlimpIrregularPastParticipleAdjectives,
    "blimp_irregular_past_participle_verbs": blimp.BlimpIrregularPastParticipleVerbs,
    "blimp_irregular_plural_subject_verb_agreement_1": blimp.BlimpIrregularPluralSubjectVerbAgreement_1,
    "blimp_irregular_plural_subject_verb_agreement_2": blimp.BlimpIrregularPluralSubjectVerbAgreement_2,
    "blimp_left_branch_island_echo_question": blimp.BlimpLeftBranchIslandEchoQuestion,
    "blimp_left_branch_island_simple_question": blimp.BlimpLeftBranchIslandSimpleQuestion,
    "blimp_matrix_question_npi_licensor_present": blimp.BlimpMatrixQuestionNpiLicensorPresent,
    "blimp_npi_present_1": blimp.BlimpNpiPresent_1,
    "blimp_npi_present_2": blimp.BlimpNpiPresent_2,
    "blimp_only_npi_licensor_present": blimp.BlimpOnlyNpiLicensorPresent,
    "blimp_only_npi_scope": blimp.BlimpOnlyNpiScope,
    "blimp_passive_1": blimp.BlimpPassive_1,
    "blimp_passive_2": blimp.BlimpPassive_2,
    "blimp_principle_A_c_command": blimp.BlimpPrinciple_ACCommand,
    "blimp_principle_A_case_1": blimp.BlimpPrinciple_ACase_1,
    "blimp_principle_A_case_2": blimp.BlimpPrinciple_ACase_2,
    "blimp_principle_A_domain_1": blimp.BlimpPrinciple_ADomain_1,
    "blimp_principle_A_domain_2": blimp.BlimpPrinciple_ADomain_2,
    "blimp_principle_A_domain_3": blimp.BlimpPrinciple_ADomain_3,
    "blimp_principle_A_reconstruction": blimp.BlimpPrinciple_AReconstruction,
    "blimp_regular_plural_subject_verb_agreement_1": blimp.BlimpRegularPluralSubjectVerbAgreement_1,
    "blimp_regular_plural_subject_verb_agreement_2": blimp.BlimpRegularPluralSubjectVerbAgreement_2,
    "blimp_sentential_negation_npi_licensor_present": blimp.BlimpSententialNegationNpiLicensorPresent,
    "blimp_sentential_negation_npi_scope": blimp.BlimpSententialNegationNpiScope,
    "blimp_sentential_subject_island": blimp.BlimpSententialSubjectIsland,
    "blimp_superlative_quantifiers_1": blimp.BlimpSuperlativeQuantifiers_1,
    "blimp_superlative_quantifiers_2": blimp.BlimpSuperlativeQuantifiers_2,
    "blimp_tough_vs_raising_1": blimp.BlimpToughVsRaising_1,
    "blimp_tough_vs_raising_2": blimp.BlimpToughVsRaising_2,
    "blimp_transitive": blimp.BlimpTransitive,
    "blimp_wh_island": blimp.BlimpWhIsland,
    "blimp_wh_questions_object_gap": blimp.BlimpWhQuestionsObjectGap,
    "blimp_wh_questions_subject_gap": blimp.BlimpWhQuestionsSubjectGap,
    "blimp_wh_questions_subject_gap_long_distance": blimp.BlimpWhQuestionsSubjectGapLongDistance,
    "blimp_wh_vs_that_no_gap": blimp.BlimpWhVsThatNoGap,
    "blimp_wh_vs_that_no_gap_long_distance": blimp.BlimpWhVsThatNoGapLongDistance,
    "blimp_wh_vs_that_with_gap": blimp.BlimpWhVsThatWithGap,
    "blimp_wh_vs_that_with_gap_long_distance": blimp.BlimpWhVsThatWithGapLongDistance,
    # Requires manual download of data.
    # "storycloze_2016": storycloze.StoryCloze2016,
    # "storycloze_2018": storycloze.StoryCloze2018,
    # "sat": sat.SATAnalogies,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
