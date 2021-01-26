#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize, pos_tag
import pickle
import re

import ade_detection.utils.localizations as loc


class DirksonNormalizer(): 
        
    #to use this function the files need to be sorted in the same folder as the script under /obj_lex/
    def load_obj(self, name):
        with open(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, loc.OBJ_LEX, f'{name}.pkl']), 'rb') as f:
            return pickle.load(f, encoding='latin1')
        

    def load_files(self): 
        self.abbr_dict = self.load_obj ('abbreviations_dict')


    def change_tup_to_list(self, tup): 
        thelist = list(tup)
        return thelist
    

    def change_list_to_tup(self,thelist): 
        tup = tuple(thelist)
        return tup
    
#---------Remove URls, email addresses and personal pronouns ------------------
        
    def replace_urls(self,list_of_msgs): 
        list_of_msgs2 = []
        for msg in list_of_msgs: 
            nw_msg = re.sub(
        r'\b' + r'((\(<{0,1}https|\(<{0,1}http|\[<{0,1}https|\[<{0,1}http|<{0,1}https|<{0,1}http)(:|;| |: )\/\/|www.)[\w\.\/#\?\=\+\;\,\&\%_\n-]+(\.[a-z]{2,4}\]{0,1}\){0,1}|\.html\]{0,1}\){0,1}|\/[\w\.\?\=#\+\;\,\&\%_-]+|[\w\/\.\?\=#\+\;\,\&\%_-]+|[0-9]+#m[0-9]+)+(\n|\b|\s|\/|\]|\)|>)',
        ' ', msg)
            list_of_msgs2.append(nw_msg)
        return list_of_msgs2    


    def replace_email(self,list_of_msgs): 
        list_of_msgs2 = []
        for msg in list_of_msgs: 
            nw_msg = re.sub (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[. ])", ' ', msg) #remove email
            nw_msg2 = re.sub (r"(@[a-zA-Z0-9]+[. ])", ' ', nw_msg) #remove usernames
            list_of_msgs2.append(nw_msg2)
        return list_of_msgs2


    def remove_empty (self,list_of_msgs): 
        check_msgs3 =[]
        for a, i in enumerate (list_of_msgs): 
            if len(i) == 0: 
                print('empty')
            else: 
                check_msgs3.append(i)
        return check_msgs3
    
    
    def remove_registered_icon (self, msg): 
        nw_msg = re.sub ('\u00AE', '', msg)
        nw_msg2 = re.sub ('\u00E9', 'e', nw_msg)
        return nw_msg2
    

    #this function has been altered because we do not wnat to remove personal pronouns
    def anonymize (self, posts): 
        posts2 = self.replace_urls (posts)
        posts3 = self.replace_email (posts2)
        posts4 = self.remove_empty(posts3)
        posts5 = [self.remove_registered_icon(p) for p in posts4]
        posts6 = [word_tokenize (sent) for sent in posts5]
        return posts6

#---------Convert to lowercase ----------------------------------------------------
    
    def lowercase (self, post):
        post1 = []
        for word in post: 
            word1 = word.lower()
            post1.append (word1)
        return post1


#---------Lexical normalization pipeline (Sarker, 2017) -------------------------------

    def loadItems(self):
        '''
        This is the primary load function.. calls other loader functions as required..
        '''    
        global english_to_american
        global noslang_dict
        global IGNORE_LIST_TRAIN
        global IGNORE_LIST

        english_to_american = {}
        IGNORE_LIST_TRAIN = []
        IGNORE_LIST = []

        english_to_american = self.loadEnglishToAmericanDict()
        noslang_dict = self.loadDictionaryData()
        for key, value in noslang_dict.items (): 
            value2 = value.lower ()
            value3 = word_tokenize (value2)
            noslang_dict[key] = value3

        return None


    def loadEnglishToAmericanDict(self):
        etoa = {}

        english = open(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, loc.OBJ_LEX, 'englishspellings.txt']))
        american = open(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, loc.OBJ_LEX,'americanspellings.txt']))
        for line in english:
            etoa[line.strip()] = american.readline().strip()
        return etoa


    def loadDictionaryData(self):
        '''
        this function loads the various dictionaries which can be used for mapping from oov to iv
        '''
        n_dict = {}
        infile = open(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, loc.OBJ_LEX,'noslang_mod.txt']))
        for line in infile:
            items = line.split(' - ')
            if len(items[0]) > 0 and len(items) > 1:
                n_dict[items[0].strip()] = items[1].strip()
        return n_dict


    #this has been changed becuase we are dealing with twitter data
    def preprocessText(self, tokens, IGNORE_LIST, ignore_username=False, ignore_hashtag=True, ignore_repeated_chars=True, eng_to_am=True, ignore_urls=False):
        '''
        Note the reason it ignores hashtags, @ etc. is because there is a preprocessing technique that is 
            designed to remove them 
        '''
        normalized_tokens =[]
        #print tokens
        text_string = ''
        # NOTE: if nesting if/else statements, be careful about execution sequence...
        for t in tokens:
            t_lower = t.strip().lower()
            # if the token is not in the IGNORE_LIST, do various transformations (e.g., ignore usernames and hashtags, english to american conversion
            # and others..
            if t_lower not in IGNORE_LIST:
                # ignore usernames '@'
                if re.match('@', t) and ignore_username:
                    IGNORE_LIST.append(t_lower)
                    text_string += t_lower + ' '
                #ignore hashtags
                elif re.match('#', t_lower) and ignore_hashtag:
                    IGNORE_LIST.append(t_lower)
                    text_string += t_lower + ' '
                #convert english spelling to american spelling
                elif t.strip().lower() in english_to_american.keys() and eng_to_am:    
                    text_string += english_to_american[t.strip().lower()] + ' '
                #URLS
                elif re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', t_lower) and ignore_urls:
                    IGNORE_LIST.append(t_lower)
                    text_string += t_lower + ' '                
                elif not ignore_repeated_chars and not re.search(r'[^a-zA-Z]', t_lower):
                    # if t_lower only contains alphabetic characters
                    t_lower = re.sub(r'([a-z])\1+', r'\1\1', t_lower)
                    text_string += t_lower + ' '  
                    # print t_lower

                # if none of the conditions match, just add the token without any changes..
                else:
                    text_string += t_lower + ' '
            else:  # i.e., if the token is in the ignorelist..
                text_string += t_lower + ' '
            normalized_tokens = text_string.split()
        # print normalized_tokens
        return normalized_tokens, IGNORE_LIST


    def dictionaryBasedNormalization(self, tokens, I_LIST, M_LIST):
        tokens2 =[]
        for t in (tokens):
            t_lower = t.strip().lower()
            if t_lower in noslang_dict.keys() and len(t_lower)>2:
                nt = noslang_dict[t_lower]
                [tokens2.append(m) for m in nt]

                if not t_lower in M_LIST:
                    M_LIST.append(t_lower)
                if not nt in M_LIST:
                    M_LIST.append(nt)
            else: 
                tokens2.append (t)
        return tokens2, I_LIST, M_LIST
    
#----Using the Sarker normalization functions ----------------------------
#Step 1 is the English normalization and step 2 is the abbreviation normalization

    def normalize_step1(self, tokens, oovoutfile=None):
        global IGNORE_LIST
        global il
        # Step 1: preprocess the text
        normalized_tokens, il = self.preprocessText(tokens, IGNORE_LIST)
        normalized_minus_ignorelist = [t for t in normalized_tokens if t not in IGNORE_LIST]
        return normalized_minus_ignorelist
    

    def normalize_step2(self, normalized_tokens, oovoutfile=None): 
        global IGNORE_LIST
        global il
        MOD_LIST = []    
        ml = MOD_LIST
        normalized_tokens, il, ml = self.dictionaryBasedNormalization(normalized_tokens, il, ml)
        return normalized_tokens


    def sarker_normalize (self,list_of_msgs): 
        self.loadItems()
        msgs_normalized = [self.normalize_step1(m) for m in list_of_msgs]
        msgs_normalized2 = [self.normalize_step2(m) for m in msgs_normalized]    
        return msgs_normalized2

#-------Domain specific abreviation expansion ----------------------------
# The list of abbreviations is input as a dictionary with tokenized output  

    def domain_specific_abbr (self, tokens, abbr): 
        post2 = [] 
        for t in tokens:
            if t in abbr.keys(): 
                nt = abbr[t]
                [post2.append(m) for m in nt]
            else: 
                post2.append(t)
        return post2


    def expand_abbr (self, data, abbr): 
        data2 = []
        for post in data: 
            post2 = self.domain_specific_abbr (tokens = post, abbr= abbr)
            data2.append(post2)
        return data2
    
#-------Spelling correction -------------------------------------------------    
    

    def run_low (self, word, voc, func, del_costs, ins_costs, sub_costs, trans_costs): 
        replacement = [' ',100]
        for token in voc: 
            sim = func(word, token, del_costs, ins_costs, sub_costs, trans_costs)
            if sim < replacement[1]:
                replacement[1] = sim
                replacement[0] = token

        return replacement   
       
    
#--------Overall normalization function--------------------------------------
    
    def normalize (self, posts): 
        self.load_files()
        posts1 = self.anonymize(posts)
        posts2 = [self.lowercase (m) for m in posts1]
        posts4 = [self.sarker_normalize(posts2)]
        posts5 = [self.expand_abbr(posts4[0], self.abbr_dict)]
        return posts5[0]
