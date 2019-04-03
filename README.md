Created By: Piyush Chandra
UIN: 671764133

How to run?
	1. Make sure to downliad the jar as it is.
	2. Run the file HW2.py
	3. If you want to test on any other file you can replace the file from the directory, for example if you want to test 
		on different relevance file, replace the file from 671764133 directory,  similarly for other files.
	3. If you see any library missing, install using pip install.
	4. All the output will be seen within 2 minutes of run.
	
The result that I have got is:

**************************************************
Average Precision for given queries:
**************************************************
Top 10:  0.18
Top 50:  0.08
Top 100:  0.05900000000000001
Top 500:  0.023

**************************************************
Average Recall for given queries:
**************************************************
Top 10:  0.1691374269005848
Top 50:  0.35211988304093567
Top 100:  0.49068713450292395
Top 500:  0.9208333333333334

Steps how I implemented code along with function names are explained below:
	1. cleanSGML - will clear all the tags and will get title and text tag inputs. 
	2. tokenizeText - will tokenize text.
	3. loadStopWords - will load stopwords.
	4. createVocabFileVocabCountDict - this is the inverted index dictionary without document frequency and using this will create 
									   inverted index dictionary with token as key and list as value. List 1st word will be count 
									   of unique documents having that token and value will be individual documents with their 
									   count of tokens in that file as a dictionary. 
	5. createFileVocabCountDict - this dictionary has maximum frequency in a document to find normalized-tf
	6. createFileVocabCountDictQuery - this is to find maximum count of any vocab in a query to find normalized tf
	7. Then we will find out tf-idf of query as well as documents.
	8. After that, we will find numerator, and 2 denominators for finding cosine similarity
	9. Then we multiply the square root of denominators and divide to find cosine similarity
	10. getKeysBasedOnDescValues - find sorted dictionary based on values
	11. getTopKResults - find top k results
	12. loadHumanRelevance - open human provided relevance document list to find precision and recall
	13. getPrecisionAndRecall - calculate precision and recall.
