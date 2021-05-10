import re

removeNL = lambda x : x.replace("\n"," ")
'''replaces newline character with spaces'''

removeSC = lambda x : re.sub('[^A-Za-z0-9 ]+', ' ', x)
'''replaces special characters with spaces'''

def assignInt4Class(x,labels):
    '''assigns Integer value to stances'''
    stances={}

    for i in range(len(labels)):
        stances.update({labels[i] : i})
    return stances[x]


def assignIntBinary(x):
    '''assigns Integer value to stances'''
    if(x=="unrelated"):
        return 0
    else:
        return 1


def preprocessDF(df_merged,labels, binaryclass = False):
    stances = {}

    for i in range(len(labels)):
        stances.update({labels[i]: i})
    '''removes newline, special characters from articlebody, headline and assigns Integer to stance'''
    df_merged["Sentence"] = df_merged["Sentence"].apply(removeNL)
    df_merged["Sentence"] = df_merged["Sentence"].apply(removeSC)

    df_merged["Claim"] = df_merged["Claim"].apply(removeNL)
    df_merged["Claim"] = df_merged["Claim"].apply(removeSC)

    if(binaryclass):
        df_merged["Stance"] = df_merged["Stance"].apply(assignIntBinary(labels))
    else:
        df_merged["Stance"] = df_merged["Stance"].apply(lambda x: stances[x])

    return df_merged,stances


def getBalancedData(ppdf,binaryclass=False):
    '''oversampling approximation for classes'''

    vc = ppdf["Stance"].value_counts()
    if(binaryclass):
        counts = [0,0]
    else:
        counts = [0,0,0,0]
    
    for index,value in vc.iteritems():
        counts[index] = value
    maxval = max(counts)
    counts = [int(maxval/x)+1 for x in counts]
    retcount = lambda x : counts[x]
    repeatSer = ppdf["Stance"].apply(retcount)
    ppdfoversampled = ppdf.reindex(ppdf.index.repeat(repeatSer))
    return ppdfoversampled

def getModelInput(str_):
    str_ = removeNL(str_)
    str_ = removeSC(str_)
    return str_