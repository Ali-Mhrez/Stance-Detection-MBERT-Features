The Arabic Fact-Checking and Stance Detection Corpus
====================================================

Description
-----------
This is a novel Arabic corpus that unifies stance detection, stance rationale, relevant document retrieval and fact checking.
The corpus contains 422 claims that are made about the war in Syria and related Middle East political issues, where each claim is labeled for “Factuality”, indicating whether they are True or False
The corpus also contains 3,042 articles that are retrieved for these claims, where each claim-article pair is annotated for “Stance”, indicating whether the article agrees, disagrees, discusses or is unrelated to the claim.
The corpus also points to which sentence(s) from the articles corresponds to the stance “rationale”.
This is the first corpus to offer such a combination.


Dataset:
--------
The dataset folder is organized as follows. Each JSON file represents a claim and contains the following fields:
* ID: The claim ID, which is used to split the data into train/test
* Claim: The claim textual content
* Fact: The factuality label of the claim (true or false)
* Article: The list of articles we retrieved for each claim using the Google Search API
* URL: the Links to each article
* Stance: The stance of each article to the claim (agree, disagree, discuss or unrelated)
* Rationale: the location of the lines in each articles that contain the rationale for agreement or disagreement with the claim.

The "train_test_splits.txt" file contains the splits that we used for the 5-fold cross validation experiments in our paper.
The split IDs correspond to the claim IDs. In other words, splitting is performed at the claim-level, i.e., a claim with all its supporting articles will appear either in the train or in the test set.


Citation:
---------
If you use this dataset in your research, kindly cite the following paper:

@InProceedings{Baly:NAACL:2018,
	title = {Integrating Stance Detection and Fact Checking in a Unified Corpus},
	author = {Baly, Ramy and Mohtarami, Mitra and Glass, James and M\`arquez, Llu\'is and Moschitti, Alessandro and Nakov, Preslav},
	booktitle = {Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics},
	series = {NAACL-HLT~'18},
	address = {New Orleans, LA, USA},
	month = {June},
	year = {2018}
}
