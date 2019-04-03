
# coding: utf-8

# In[ ]:


#Assignment 2
#Submitted by: Piyush Chandra
#UIN: 671764133


# In[1]:


#this is to clean all the sgml tags and get string from title and text tag
def cleanSGML(input):
    import re
    temp = re.sub('\s+',' ',input)
    text1=re.findall("<TITLE>(.*?)</TITLE>", temp)
    text2=re.findall("<TEXT>(.*?)</TEXT>", temp)
    text1=''.join(text1)
    text2=''.join(text2)
    return ''.join([text1, text2])


# In[2]:


# this method will lowercase each word , remove punctuation and tokenize each word
def tokenizeText(temp):
    import re
    temp=temp.lower()
    temp_no_num=re.sub(r'\b[0-9]+\b\s*','',temp) #remove numbers
    temp_remove_less_3_char=re.sub(r'\b[a-z]{1,2}\b','',temp_no_num) #remove char of 1 or 2 size
    temp_no_punc=re.sub(r'[^\w\s]','',temp_remove_less_3_char) #remove punctuations
    return temp_no_punc.split() #tokenize all words


# In[21]:


def loadStopWords():
    with open('stopwords.txt', 'r') as stopFile:
        stops = stopFile.read().split('\n')
    return stops


# In[24]:


#this is the inverted index dictionary without document frequency
def createVocabFileVocabCountDict(path):
    import glob, re
    from nltk import PorterStemmer
    vocab_dict = {}
    count=0
    totalDocs=0
    stopWordList = loadStopWords()
    for file in glob.glob(path):
        totalDocs+=1
        with open(file, 'r') as fin:
            fileName=str(file).lstrip('cranfieldDocs\\').lstrip('0')
            temp=fin.read()
            temp_sgml = cleanSGML(temp)
            temp_tokenized=tokenizeText(temp_sgml)
            for word in temp_tokenized:
                if word in stopWordList:
                    continue
                else:
                    stem_word = PorterStemmer().stem(word)
                    if stem_word in stopWordList:
                        continue
                    else:
                        fileNameVocabCountDict={}
                        if stem_word in vocab_dict:
                            fileNameVocabCountDict=vocab_dict.get(stem_word)
                            if(fileName in fileNameVocabCountDict):
                                count = fileNameVocabCountDict.get(fileName)
                                count+=1
                                fileNameVocabCountDict[fileName]=count
                                vocab_dict[stem_word] = fileNameVocabCountDict
                            else:
                                fileNameVocabCountDict[fileName]=1
                                vocab_dict[stem_word] = fileNameVocabCountDict
                        else:
                            fileNameVocabCountDict[fileName]=1
                            vocab_dict[stem_word] = fileNameVocabCountDict
    return vocab_dict, totalDocs


# In[27]:


import os
path=os.path.join("cranfieldDocs","*")
vocab_doc_vocab_count_dict, totalDocs = createVocabFileVocabCountDict(path)


# In[6]:


# this is the final inverted index with document frequency
invertedIndexDict={}
for key, val in vocab_doc_vocab_count_dict.items():
    invertedIndexDict[key]=[len(val),val]


# In[28]:


# this dictionary will help to build maximum frequency in a document to find normalized-tf
def createFileVocabCountDict(path):
    import glob, re
    from nltk import PorterStemmer
    file_dict = {}
    count=0
    stopWordList = loadStopWords()
    for file in glob.glob(path):
        with open(file, 'r') as fin:
            temp=fin.read()
            fileName=str(file).lstrip('cranfieldDocs\\').lstrip('0')
            temp_sgml = cleanSGML(temp)
            temp_tokenized=tokenizeText(temp_sgml)
            for word in temp_tokenized:
                if word in stopWordList:
                    continue
                else:
                    stem_word = PorterStemmer().stem(word)
                    if stem_word in stopWordList:
                        continue
                    else:
                        vocabCountDict={}
                        if fileName in file_dict:
                            vocabCountDict=file_dict.get(fileName)
                            if(stem_word in vocabCountDict):
                                count = vocabCountDict.get(stem_word)
                                count+=1
                                vocabCountDict[stem_word]=count
                                file_dict[fileName] = vocabCountDict
                            else:
                                vocabCountDict[stem_word]=1
                                file_dict[fileName] = vocabCountDict
                        else:
                            vocabCountDict[stem_word]=1
                            file_dict[fileName] = vocabCountDict
    return file_dict

path=os.path.join("cranfieldDocs","*")
file_dict=createFileVocabCountDict(path)


# In[8]:


# this dictionary has maximum frequency in a document to find normalized-tf
max_freq_of_each_doc_dict = {}
for key, value in file_dict.items():
        max_freq_of_each_doc_dict[key]=max(value.values())


# In[31]:


# this is to find maximum count of any vocab in a query to find normalized tf
def createFileVocabCountDictQuery(path):
    import glob, re
    from nltk import PorterStemmer
    file_dict = {}
    stopWordList = loadStopWords()
    count=0;
    queryNum=0
    file = open(path, "r")
    for line in file:
        queryNum+=1
        temp_tokenized=tokenizeText(line)
        weight=0
        for word in temp_tokenized:
                if word in stopWordList:
                    continue
                else:
                    stem_word = PorterStemmer().stem(word)
                    if stem_word in stopWordList:
                        continue
                    else:
                        vocabCountDict={}
                        if queryNum in file_dict:
                            vocabCountDict=file_dict.get(queryNum)
                            if(stem_word in vocabCountDict):
                                count = vocabCountDict.get(stem_word)
                                count+=1
                                vocabCountDict[stem_word]=count
                                file_dict[queryNum] = vocabCountDict
                            else:
                                vocabCountDict[stem_word]=1
                                file_dict[queryNum] = vocabCountDict
                        else:
                            vocabCountDict[stem_word]=1
                            file_dict[queryNum] = vocabCountDict
        
    return file_dict



file="queries.txt"
file_dict_query =createFileVocabCountDictQuery(file)


# In[10]:


max_freq_of_each_doc_dict_query = {}
for key, value in file_dict_query.items():
        max_freq_of_each_doc_dict_query[key]=max(value.values())


# In[11]:


# this is tf-idf of each vocab in a query, if term not present in inverted index, then we won't find tf-idf as it will be 0
import math
queryTfIdfDict={}
for key, value in file_dict_query.items():
    vocabTfIdf={}
    for key2,value2 in value.items():
        if key2 in invertedIndexDict:
            tf=value2/max_freq_of_each_doc_dict_query.get(key)
            idf= math.log((totalDocs/invertedIndexDict.get(key2)[0]),2)
            if tf*idf!=0:
                vocabTfIdf[key2]=tf*idf
    queryTfIdfDict[key]=vocabTfIdf


# In[12]:


# tf-idf of each document tokens
tokenDocTfIdfDict={}
for token, value in invertedIndexDict.items():
    docIdVocabCount=invertedIndexDict.get(token)[1]
    docIdTfIdfDict={}
    for docId, tokenCount in docIdVocabCount.items():
        tf=tokenCount/max_freq_of_each_doc_dict.get(docId)
        idf=math.log((totalDocs/invertedIndexDict.get(token)[0]),2)
        docIdTfIdfDict[docId]=tf*idf
    tokenDocTfIdfDict[token]=docIdTfIdfDict


# In[13]:


# rearrange the dictionary
docTfIdfDict2={}
for docId, vocabTfIdfDict in tokenDocTfIdfDict.items():
    for vocab, tfIdf in vocabTfIdfDict.items():
        if vocab in docTfIdfDict2:
            temp=docTfIdfDict2.get(vocab)
            temp[docId]=tfIdf
            docTfIdfDict2[vocab]=temp
        else:
            temp={docId:tfIdf}
            docTfIdfDict2[vocab]=temp


# In[14]:


# now we will find numerator, and denominator for finding cosine similarity
docIdNumOfCosineSimDict={}
queryCumulDenomDict={}
docCumulDenomDict={}
for queryIdKey, queryVocabTfIdfDict in queryTfIdfDict.items():
    docNumeratorDict={}
    docDenomDict={}
    queryDenomDict={}
    for queryToken, queryTokenTfIdf in queryVocabTfIdfDict.items():
        for docId, docVocabTfIdf in docTfIdfDict2.items():
            for docVocab, docTdfIdf in docVocabTfIdf.items():
                if queryToken==docVocab:
                    if docId in docNumeratorDict:
                        docNumeratorDict[docId]+=docTdfIdf*queryTokenTfIdf
                        docDenomDict[docId]+=docTdfIdf*docTdfIdf
                        queryDenomDict[docId]+=queryTokenTfIdf*queryTokenTfIdf
                    else:
                        docNumeratorDict[docId]=docTdfIdf*queryTokenTfIdf
                        docDenomDict[docId]=docTdfIdf*docTdfIdf
                        queryDenomDict[docId]=queryTokenTfIdf*queryTokenTfIdf
    docIdNumOfCosineSimDict[queryIdKey]=docNumeratorDict
    docCumulDenomDict[queryIdKey]=docDenomDict
    queryCumulDenomDict[queryIdKey]=queryDenomDict


# In[15]:


# now we will multiply the square root of denominators and divide to find cosine similarity
queryDocCosineSimDict={}
queryDocCosSimDict={}
for queryId, docWeightDict in docIdNumOfCosineSimDict.items():
    tempDict={}
    tempDict2={}
    for docId, weight in docWeightDict.items():
        tempDict[docId] = weight
        temp=(queryCumulDenomDict.get(queryId).get(docId)**(.5))*(docCumulDenomDict.get(queryId).get(docId)**(.5))
        tempDict2[docId] = weight/temp
    queryDocCosSimDict[queryId]=tempDict2
    queryDocCosineSimDict[queryId]=tempDict


# In[16]:


# find sorted dictionary based on values
def getKeysBasedOnDescValues(vocab_dict):
    return sorted(vocab_dict, key=vocab_dict.get, reverse=True)

systemRelDict={}
for queryId, cumulWeightOfFileDict in queryDocCosineSimDict.items():
    topKResults=getKeysBasedOnDescValues(cumulWeightOfFileDict)
    systemRelDict[queryId]= topKResults


# In[17]:


# find top k results
def getTopKResults(k):
    sortedDict={}
    for queryId, docIdList in systemRelDict.items():
        sortedDict[queryId]=docIdList[:k]
    return sortedDict

top10SysDocs=getTopKResults(10)
top50SysDocs=getTopKResults(50)
top100SysDocs=getTopKResults(100)
top500SysDocs=getTopKResults(500)


# In[18]:


# open human provided relevance document list to find precision and recall
def loadHumanRelevance():
    with open('relevance.txt', 'r') as relFile:
        queryRel = relFile.read().split('\n')
    humanRelDict={}
    a=100
    for line in queryRel:
        temp=line.split()
        if int(temp[0]) in humanRelDict:
            tempList=humanRelDict[int(temp[0])]
            tempList.append(temp[1])
            humanRelDict[int(temp[0])]=tempList
        else:
            humanRelDict[int(temp[0])] = [temp[1]]
    return humanRelDict

humanRelDict=loadHumanRelevance()


# In[19]:


# calculate precision and recall
def getPrecisionAndRecall(topKDocuments):
    count=0
    totalPrecisionOfQuery=0
    totalRecallOfQuery =0
    for queryId, fileNameList in humanRelDict.items():
        docCount=0
        for fileName in fileNameList:
            sysGenDocs=topKDocuments.get(queryId)
            if fileName in sysGenDocs:
                docCount+=1
        totalPrecisionOfQuery+=docCount/len(topKDocuments.get(queryId))
        totalRecallOfQuery+=docCount/len(fileNameList)
    return totalPrecisionOfQuery/len(max_freq_of_each_doc_dict_query), totalRecallOfQuery/len(max_freq_of_each_doc_dict_query)


# In[20]:


#print all the precision and recall
top10Precision, top10Recall = getPrecisionAndRecall(top10SysDocs)
top50Precision, top50Recall = getPrecisionAndRecall(top50SysDocs)
top100Precision, top100Recall = getPrecisionAndRecall(top100SysDocs)
top500Precision, top500Recall = getPrecisionAndRecall(top500SysDocs)


print('**************************************************')
print('Average Precision for given queries:')
print('**************************************************')
print('Top 10: ', top10Precision)
print('Top 50: ', top50Precision)
print('Top 100: ', top100Precision)
print('Top 500: ', top500Precision)
print('')
print('**************************************************')
print('Average Recall for given queries:')
print('**************************************************')
print('Top 10: ', top10Recall)
print('Top 50: ', top50Recall)
print('Top 100: ', top100Recall)
print('Top 500: ', top500Recall)

