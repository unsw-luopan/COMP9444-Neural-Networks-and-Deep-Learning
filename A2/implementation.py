from collections import Counter

import tensorflow as tf
import re
from tensorflow.contrib import rnn

BATCH_SIZE = 64
MAX_WORDS_IN_REVIEW = 300  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set(
    {"'d", "'ll", "'m", "'re", "'s", "'t", "'ve", 'ZT', 'ZZ', 'a', "a's", 'able', 'about', 'above', 'abst',
     'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'adopted', 'affected',
     'affecting', 'affects', 'after', 'afterwards', 'again', 'against', 'ah', "ain't", 'all', 'allow', 'allows',
     'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and',
     'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways',
     'anywhere', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'are', 'area', 'areas',
     'aren', "aren't", 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asked', 'asking', 'asks', 'associated', 'at',
     'auth', 'available', 'away', 'awfully', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because',
     'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'beginnings',
     'begins', 'behind', 'being', 'beings', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between',
     'beyond', 'big', 'biol', 'both', 'brief', 'briefly', 'brother', 'but', 'by', 'c', "c'mon", "c's", 'ca', 'came',
     'can',
     "can't", 'cannot', 'cant', 'case', 'cases', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clear',
     'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain',
     'containing', 'contains', 'corresponding', 'could', "couldn't", 'couldnt', 'course', 'currently', 'd', 'date',
     'definitely', 'describe', 'described', 'despite', 'did', "didn't", 'differ', 'different', 'differently', 'discuss',
     'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downed', 'downing', 'downs', 'downwards', 'due',
     'during', 'e', 'each', 'early', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere',
     'end', 'ended', 'ending', 'ends', 'enough', 'entirely', 'especially', 'et', 'et-al', 'etc', 'even', 'evenly',
     'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f',
     'face', 'faces', 'fact', 'father', 'facts', 'far', 'felt', 'few', 'ff', 'fifth', 'find', 'finds', 'first', 'five',
     'fix',
     'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'found', 'four', 'from', 'full', 'fully',
     'further', 'furthered', 'furthering', 'furthermore', 'furthers', 'g', 'gave', 'general', 'generally', 'get',
     'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'going', 'gone', 'good', 'goods', 'got',
     'gotten', 'great', 'greater', 'greatest', 'greetings', 'group', 'grouped', 'grouping', 'groups', 'h', 'had',
     "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hed', 'hello', 'help',
     'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes',
     'hi', 'hid', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'hither', 'home', 'hopefully', 'how', 'howbeit',
     'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'id', 'ie', 'if', 'ignored', 'im', 'immediate',
     'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'include', 'indeed', 'index', 'indicate',
     'indicated', 'indicates', 'information', 'inner', 'insofar', 'instead', 'interest', 'interested', 'interesting',
     'interests', 'into', 'invention', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'itd', 'its', 'itself',
     'j', 'just', 'k', 'keep', 'keeps', 'kept', 'keys', 'kg', 'kind', 'km', 'knew', 'know', 'known', 'knows', 'l',
     'large', 'largely', 'last', 'lately', 'later', 'latest', 'latter', 'latterly', 'least', 'less', 'lest', 'let',
     "let's", 'lets', 'like', 'liked', 'likely', 'line', 'little', 'long', 'longer', 'longest', 'look', 'looking',
     'looks', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', 'me', 'mean',
     'means', 'meantime', 'meanwhile', 'member', 'members', 'men', 'merely', 'mg', 'might', 'million', 'miss', 'ml',
     'more', 'moreover', 'most', 'mother', 'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', "n't",
     'na', 'name',
     'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needed', 'needing', 'needs',
     'neither', 'never', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'nobody', 'non',
     'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'novel', 'now', 'nowhere',
     'number', 'numbers', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old',
     'older', 'oldest', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'open', 'opened', 'opening', 'opens',
     'or', 'ord', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'otherwise', 'ought', 'our', 'ours',
     'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page', 'pages', 'part', 'parted',
     'particular', 'particularly', 'parting', 'parts', 'past', 'per', 'perhaps', 'place', 'placed', 'places', 'please',
     'plus', 'point', 'pointed', 'pointing', 'points', 'poorly', 'possible', 'possibly', 'potentially', 'pp',
     'predominantly', 'present', 'presented', 'presenting', 'presents', 'presumably', 'previously', 'primarily',
     'probably', 'problem', 'problems', 'promptly', 'proud', 'provides', 'put', 'puts', 'q', 'que', 'quickly', 'quite',
     'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs',
     'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting',
     'results', 'right', 'room', 'rooms', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec', 'second',
     'secondly', 'seconds', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'sees', 'self',
     'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', "she'll", 'shed', 'shes',
     'should', "shouldn't", 'show', 'showed', 'showing', 'shown', 'showns', 'shows', 'side', 'sides', 'significant',
     'significantly', 'similar', 'similarly', 'since', 'six', 'slightly', 'small', 'smaller', 'smallest', 'so', 'some',
     'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere',
     'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'state', 'states', 'still', 'stop',
     'strongly', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure', 't', "t's",
     'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's",
     "that've", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'll",
     "there's", "there've", 'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'thereof', 'therere', 'theres',
     'thereto', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'theyd', 'theyre', 'thing',
     'things', 'think', 'thinks', 'third', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh',
     'thought', 'thoughts', 'thousand', 'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to',
     'today', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'turn',
     'turned', 'turning', 'turns', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely',
     'until', 'unto', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using',
     'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wanted',
     'wanting', 'wants', 'was', "wasn't", 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'wed', 'welcome',
     'well', 'wells', 'went', 'were', "weren't", 'what', "what'll", "what's", 'whatever', 'whats', 'when', 'whence',
     'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever',
     'whether', 'which', 'while', 'whim', 'whither', 'who', "who'll", "who's", 'whod', 'whoever', 'whole', 'whom',
     'whomever', 'whos', 'whose', 'why', 'widely', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't",
     'wonder', 'words', 'work', 'worked', 'working', 'works', 'world', 'would', "wouldn't", 'www', 'x', 'y', 'year',
     'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'young', 'younger', 'youngest',
     'your', 'youre', 'yours', 'yourself', 'yourselves', 'z', 'zero', 'zt', 'zz'
     })


def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # lower words
    review = review.lower()
    review = review.split()
    # remove stop words
    review = [w for w in review if w not in stop_words]
    r1 = ' '.join(review)
    r2 = ''.join(r1)
    # remove puncations
    del_content = re.sub(r"[\s\n\d#=||&:?<>}{@+\.\!\/.,$%^*_\---(+)\"\']+|[+——！，。？?、\[~@=《》：#；：:\]’“”‘￥%……&*（）]+]",
                         '\n', r2)
    # remove number
    result_list = re.findall('[a-z A-Z]+', del_content)
    # remove too short words
    r3 = [w for w in result_list if len(w) > 2]
    processed_review = [w for w in r3 if w not in stop_words]
    #r5 = ' '.join(r4)
    #r6 = r5.split()
    #print(r5)
    word_counts = Counter(processed_review)
    #print(word_counts)
    processed_review = [word for word in word_counts if word_counts[word] < 5]
    #print(processed_review)
    # processed_review = ' '.join(processed_review)
    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    lstmsize = 64
    numClasses = 2
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, numClasses], name="labels")
    # build lstm
    lstm_cell_1 = rnn.BasicLSTMCell(num_units=lstmsize, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell_2 = rnn.BasicLSTMCell(num_units=lstmsize, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell_3 = rnn.BasicLSTMCell(num_units=lstmsize, forget_bias=1.0, state_is_tuple=True)
    # drop out
    lstm_cell_1 = rnn.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=0.6)
    #lstm_cell_2 = rnn.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=0.5)
    #lstm_cell_3 = rnn.DropoutWrapper(cell=lstm_cell_3, output_keep_prob=0.5)
    # link several lstm layer
    #lstm_cell = rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2, lstm_cell_3])
    dropout_keep_prob = tf.placeholder(tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell_1, input_data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([lstmsize, numClasses]))
    bias = tf.Variable(tf.truncated_normal([numClasses]))
    # reshape outputs
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # calculate accuracy
    Accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels), name="loss")
    # apply learning rate decay
    #global_step = 100000
    #learning_rate = tf.train.exponential_decay(learning_rate=0.005, global_step=global_step, decay_steps=1000,
    #                                           decay_rate=0.96)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
