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
from lm_eval.base import Task, rf 
from lm_eval.metrics import mean 

_CITATION = """@misc{https://doi.org/10.48550/arxiv.2206.04615,
  doi = {10.48550/ARXIV.2206.04615},
  
  url = {https://arxiv.org/abs/2206.04615},
  
  author = {Srivastava, Aarohi and Rastogi, Abhinav and Rao, Abhishek and Shoeb, Abu Awal Md and Abid, Abubakar and Fisch, Adam and Brown, Adam R. and Santoro, Adam and Gupta, Aditya and Garriga-Alonso, Adrià and Kluska, Agnieszka and Lewkowycz, Aitor and Agarwal, Akshat and Power, Alethea and Ray, Alex and Warstadt, Alex and Kocurek, Alexander W. and Safaya, Ali and Tazarv, Ali and Xiang, Alice and Parrish, Alicia and Nie, Allen and Hussain, Aman and Askell, Amanda and Dsouza, Amanda and Slone, Ambrose and Rahane, Ameet and Iyer, Anantharaman S. and Andreassen, Anders and Madotto, Andrea and Santilli, Andrea and Stuhlmüller, Andreas and Dai, Andrew and La, Andrew and Lampinen, Andrew and Zou, Andy and Jiang, Angela and Chen, Angelica and Vuong, Anh and Gupta, Animesh and Gottardi, Anna and Norelli, Antonio and Venkatesh, Anu and Gholamidavoodi, Arash and Tabassum, Arfa and Menezes, Arul and Kirubarajan, Arun and Mullokandov, Asher and Sabharwal, Ashish and Herrick, Austin and Efrat, Avia and Erdem, Aykut and Karakaş, Ayla and Roberts, B. Ryan and Loe, Bao Sheng and Zoph, Barret and Bojanowski, Bartłomiej and Özyurt, Batuhan and Hedayatnia, Behnam and Neyshabur, Behnam and Inden, Benjamin and Stein, Benno and Ekmekci, Berk and Lin, Bill Yuchen and Howald, Blake and Diao, Cameron and Dour, Cameron and Stinson, Catherine and Argueta, Cedrick and Ramírez, César Ferri and Singh, Chandan and Rathkopf, Charles and Meng, Chenlin and Baral, Chitta and Wu, Chiyu and Callison-Burch, Chris and Waites, Chris and Voigt, Christian and Manning, Christopher D. and Potts, Christopher and Ramirez, Cindy and Rivera, Clara E. and Siro, Clemencia and Raffel, Colin and Ashcraft, Courtney and Garbacea, Cristina and Sileo, Damien and Garrette, Dan and Hendrycks, Dan and Kilman, Dan and Roth, Dan and Freeman, Daniel and Khashabi, Daniel and Levy, Daniel and González, Daniel Moseguí and Perszyk, Danielle and Hernandez, Danny and Chen, Danqi and Ippolito, Daphne and Gilboa, Dar and Dohan, David and Drakard, David and Jurgens, David and Datta, Debajyoti and Ganguli, Deep and Emelin, Denis and Kleyko, Denis and Yuret, Deniz and Chen, Derek and Tam, Derek and Hupkes, Dieuwke and Misra, Diganta and Buzan, Dilyar and Mollo, Dimitri Coelho and Yang, Diyi and Lee, Dong-Ho and Shutova, Ekaterina and Cubuk, Ekin Dogus and Segal, Elad and Hagerman, Eleanor and Barnes, Elizabeth and Donoway, Elizabeth and Pavlick, Ellie and Rodola, Emanuele and Lam, Emma and Chu, Eric and Tang, Eric and Erdem, Erkut and Chang, Ernie and Chi, Ethan A. and Dyer, Ethan and Jerzak, Ethan and Kim, Ethan and Manyasi, Eunice Engefu and Zheltonozhskii, Evgenii and Xia, Fanyue and Siar, Fatemeh and Martínez-Plumed, Fernando and Happé, Francesca and Chollet, Francois and Rong, Frieda and Mishra, Gaurav and Winata, Genta Indra and de Melo, Gerard and Kruszewski, Germán and Parascandolo, Giambattista and Mariani, Giorgio and Wang, Gloria and Jaimovitch-López, Gonzalo and Betz, Gregor and Gur-Ari, Guy and Galijasevic, Hana and Kim, Hannah and Rashkin, Hannah and Hajishirzi, Hannaneh and Mehta, Harsh and Bogar, Hayden and Shevlin, Henry and Schütze, Hinrich and Yakura, Hiromu and Zhang, Hongming and Wong, Hugh Mee and Ng, Ian and Noble, Isaac and Jumelet, Jaap and Geissinger, Jack and Kernion, Jackson and Hilton, Jacob and Lee, Jaehoon and Fisac, Jaime Fernández and Simon, James B. and Koppel, James and Zheng, James and Zou, James and Kocoń, Jan and Thompson, Jana and Kaplan, Jared and Radom, Jarema and Sohl-Dickstein, Jascha and Phang, Jason and Wei, Jason and Yosinski, Jason and Novikova, Jekaterina and Bosscher, Jelle and Marsh, Jennifer and Kim, Jeremy and Taal, Jeroen and Engel, Jesse and Alabi, Jesujoba and Xu, Jiacheng and Song, Jiaming and Tang, Jillian and Waweru, Joan and Burden, John and Miller, John and Balis, John U. and Berant, Jonathan and Frohberg, Jörg and Rozen, Jos and Hernandez-Orallo, Jose and Boudeman, Joseph and Jones, Joseph and Tenenbaum, Joshua B. and Rule, Joshua S. and Chua, Joyce and Kanclerz, Kamil and Livescu, Karen and Krauth, Karl and Gopalakrishnan, Karthik and Ignatyeva, Katerina and Markert, Katja and Dhole, Kaustubh D. and Gimpel, Kevin and Omondi, Kevin and Mathewson, Kory and Chiafullo, Kristen and Shkaruta, Ksenia and Shridhar, Kumar and McDonell, Kyle and Richardson, Kyle and Reynolds, Laria and Gao, Leo and Zhang, Li and Dugan, Liam and Qin, Lianhui and Contreras-Ochando, Lidia and Morency, Louis-Philippe and Moschella, Luca and Lam, Lucas and Noble, Lucy and Schmidt, Ludwig and He, Luheng and Colón, Luis Oliveros and Metz, Luke and Şenel, Lütfi Kerem and Bosma, Maarten and Sap, Maarten and ter Hoeve, Maartje and Farooqi, Maheen and Faruqui, Manaal and Mazeika, Mantas and Baturan, Marco and Marelli, Marco and Maru, Marco and Quintana, Maria Jose Ramírez and Tolkiehn, Marie and Giulianelli, Mario and Lewis, Martha and Potthast, Martin and Leavitt, Matthew L. and Hagen, Matthias and Schubert, Mátyás and Baitemirova, Medina Orduna and Arnaud, Melody and McElrath, Melvin and Yee, Michael A. and Cohen, Michael and Gu, Michael and Ivanitskiy, Michael and Starritt, Michael and Strube, Michael and Swędrowski, Michał and Bevilacqua, Michele and Yasunaga, Michihiro and Kale, Mihir and Cain, Mike and Xu, Mimee and Suzgun, Mirac and Tiwari, Mo and Bansal, Mohit and Aminnaseri, Moin and Geva, Mor and Gheini, Mozhdeh and T, Mukund Varma and Peng, Nanyun and Chi, Nathan and Lee, Nayeon and Krakover, Neta Gur-Ari and Cameron, Nicholas and Roberts, Nicholas and Doiron, Nick and Nangia, Nikita and Deckers, Niklas and Muennighoff, Niklas and Keskar, Nitish Shirish and Iyer, Niveditha S. and Constant, Noah and Fiedel, Noah and Wen, Nuan and Zhang, Oliver and Agha, Omar and Elbaghdadi, Omar and Levy, Omer and Evans, Owain and Casares, Pablo Antonio Moreno and Doshi, Parth and Fung, Pascale and Liang, Paul Pu and Vicol, Paul and Alipoormolabashi, Pegah and Liao, Peiyuan and Liang, Percy and Chang, Peter and Eckersley, Peter and Htut, Phu Mon and Hwang, Pinyu and Miłkowski, Piotr and Patil, Piyush and Pezeshkpour, Pouya and Oli, Priti and Mei, Qiaozhu and Lyu, Qing and Chen, Qinlang and Banjade, Rabin and Rudolph, Rachel Etta and Gabriel, Raefer and Habacker, Rahel and Delgado, Ramón Risco and Millière, Raphaël and Garg, Rhythm and Barnes, Richard and Saurous, Rif A. and Arakawa, Riku and Raymaekers, Robbe and Frank, Robert and Sikand, Rohan and Novak, Roman and Sitelew, Roman and LeBras, Ronan and Liu, Rosanne and Jacobs, Rowan and Zhang, Rui and Salakhutdinov, Ruslan and Chi, Ryan and Lee, Ryan and Stovall, Ryan and Teehan, Ryan and Yang, Rylan and Singh, Sahib and Mohammad, Saif M. and Anand, Sajant and Dillavou, Sam and Shleifer, Sam and Wiseman, Sam and Gruetter, Samuel and Bowman, Samuel R. and Schoenholz, Samuel S. and Han, Sanghyun and Kwatra, Sanjeev and Rous, Sarah A. and Ghazarian, Sarik and Ghosh, Sayan and Casey, Sean and Bischoff, Sebastian and Gehrmann, Sebastian and Schuster, Sebastian and Sadeghi, Sepideh and Hamdan, Shadi and Zhou, Sharon and Srivastava, Shashank and Shi, Sherry and Singh, Shikhar and Asaadi, Shima and Gu, Shixiang Shane and Pachchigar, Shubh and Toshniwal, Shubham and Upadhyay, Shyam and Shyamolima,  and {Debnath} and Shakeri, Siamak and Thormeyer, Simon and Melzi, Simone and Reddy, Siva and Makini, Sneha Priscilla and Lee, Soo-Hwan and Torene, Spencer and Hatwar, Sriharsha and Dehaene, Stanislas and Divic, Stefan and Ermon, Stefano and Biderman, Stella and Lin, Stephanie and Prasad, Stephen and Piantadosi, Steven T. and Shieber, Stuart M. and Misherghi, Summer and Kiritchenko, Svetlana and Mishra, Swaroop and Linzen, Tal and Schuster, Tal and Li, Tao and Yu, Tao and Ali, Tariq and Hashimoto, Tatsu and Wu, Te-Lin and Desbordes, Théo and Rothschild, Theodore and Phan, Thomas and Wang, Tianle and Nkinyili, Tiberius and Schick, Timo and Kornev, Timofei and Telleen-Lawton, Timothy and Tunduny, Titus and Gerstenberg, Tobias and Chang, Trenton and Neeraj, Trishala and Khot, Tushar and Shultz, Tyler and Shaham, Uri and Misra, Vedant and Demberg, Vera and Nyamai, Victoria and Raunak, Vikas and Ramasesh, Vinay and Prabhu, Vinay Uday and Padmakumar, Vishakh and Srikumar, Vivek and Fedus, William and Saunders, William and Zhang, William and Vossen, Wout and Ren, Xiang and Tong, Xiaoyu and Zhao, Xinran and Wu, Xinyi and Shen, Xudong and Yaghoobzadeh, Yadollah and Lakretz, Yair and Song, Yangqiu and Bahri, Yasaman and Choi, Yejin and Yang, Yichi and Hao, Yiding and Chen, Yifu and Belinkov, Yonatan and Hou, Yu and Hou, Yufang and Bai, Yuntao and Seid, Zachary and Zhao, Zhuoye and Wang, Zijian and Wang, Zijie J. and Wang, Zirui and Wu, Ziyi},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

"""


class AutoCategroization(Task):
    VERSION = "0.0.1"
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "bigbench"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "auto_categorization"

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
        query = doc['inputs']
        targets = doc['targets']
        return {
            "query": query,  # The query prompt.
            "targets": targets, # the answer(s)
        }

    def doc_to_text(self, doc):
        # default? might not work...
        return doc['query']

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        print(doc.keys(), doc)
        target = doc['targets']
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
        # TODO: Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        #stolen from gsmk8k.py
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
        targets = doc['targets']
        gen_answer = results[0]

        gen_answer = gen_answer.lower().strip() 
        acc = 0
        #TODO: replcae with something like bleu? or add it? 
        if isinstance(targets, list): 
            for target in targets: 
                if gen_answer == target.lower().strip(): 
                    acc = 1
        else: 
            target = targets
            if gen_answer == target.lower().strip(): 
                    acc = 1

        return {'acc':acc}

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
        return {"acc": mean}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {'acc': True}
