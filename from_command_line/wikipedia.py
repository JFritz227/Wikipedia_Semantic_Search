# run from command line using the command ```$ python wikipedia.py```

# import packages
import pymongo
import re
import requests
import pandas as pd
import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import sys

# use IP address for database location
client = pymongo.MongoClient('54.200.30.7', 27016)

# functions class with functions to clean text, get page ids for each wikipedia page, find title, strip punctuation from documents except for periods, 
# delete periods, strip html, and get page contents
class functions():
    
    def clean_text(self, string):
        string = functions().delete_periods(string)
        new_string = string.replace(' ', '_')
        new_string = new_string.lower()
        return new_string
    
    def pageids(self, category):
        category = category.replace(' ', '+')
        category = category.lower()
        r = requests.get('http://en.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category%3A+' + category + '&cmlimit=max')
        pageids = pd.DataFrame(r.json()['query']['categorymembers'])['pageid']
        return pageids

    def find_title(self, page_number):
        query = 'http://en.wikipedia.org/w/api.php?action=query&prop=extracts&\
             rvprop=content&rvsection=0&format=json&pageids={}'.format(page_number)
        my_request = requests.get(query).json()
        return my_request['query']['pages'][str(page_number)]['title']

    def strip_punctuation(self, s):
        return ''.join(c for c in s if c not in punctuation or c is '.')
    
    def delete_periods(self, s):
        return ''.join(c for c in s if c not in punctuation)

    def striphtml(self, data):
        p = re.compile(r'<.*?>')
        return p.sub('', data)

    def get_page_contents(self, pageid, category):
        query = 'http://en.wikipedia.org/w/api.php?action=query&prop=extracts&\
             rvprop=content&rvsection=0&format=json&pageids={}'.format(pageid)
        my_request = requests.get(query).json()
        title = self.striphtml(my_request['query']['pages'][str(pageid)]['title']) 
        no_html_string = self.striphtml(my_request['query']['pages'][str(pageid)]['extract']).replace('\n', ' ')
        return {'title':title, 'raw page text':self.strip_punctuation(no_html_string), 'category':category}
    
# create class to add database, drop database and dive into the wikipedia categories
class mongodb():
    wiki_db = []
    database_name_ = ''
    
    def __init__(self, database_name):
        function_init = functions()
        self.database_name_ = function_init.clean_text(database_name)
        
    def collections(self):
        return client[self.database_name_].collection_names()
        
    def add_database(self):
        self.wiki_db = client[self.database_name_]
        
    def drop_database(self):
        client.drop_database(self.database_name_)
    
    def add_first_layer(self, database_name):
        function_init = functions()
        database_edited = function_init.clean_text(database_name)
        wiki_level_0 = self.wiki_db[database_edited]
        wiki_level_1 = self.wiki_db.subcategories
        these_pageids = function_init.pageids(self.database_name_)
        for pages in these_pageids:
            if 'Category:' not in function_init.find_title(pages):
                wiki_level_0.insert_one(function_init.get_page_contents(pages, self.database_name_))
            if 'Category:' in function_init.find_title(pages) :
                subcategory = function_init.find_title(pages)
                subcategory = subcategory.replace('Category:', database_name + ' -> ')
                wiki_level_1.insert_one(function_init.get_page_contents(pages, subcategory))
                
    def add_second_layer(self, database_name, subcategory_name):
        function_init = functions()
        database_edited = function_init.clean_text(database_name)
        subcat_edited = function_init.clean_text(subcategory_name)
        wiki_level_0 = self.wiki_db[database_edited][subcat_edited]
        wiki_level_1 = self.wiki_db[database_edited][subcat_edited].subcategories
        these_pageids = functions().pageids(subcategory_name)
        for pages in these_pageids:
            if 'Category:' not in functions().find_title(pages):
                wiki_level_0.insert_one(functions().get_page_contents(pages, 
                                                                      database_name + ' -> ' + subcategory_name))
            if 'Category:' in functions().find_title(pages): 
                subcategory = functions().find_title(pages)
                subcategory = subcategory.replace('Category:', 
                                                  database_name + ' -> ' + subcategory_name + ' -> ')
                wiki_level_1.insert_one(functions().get_page_contents(pages, subcategory))
                
    def add_third_layer(self, database_name, subcategory_name_0, subcategory_name_1):
        function_init = functions()
        subcategory_edited_0 = function_init.clean_text(subcategory_name_0)
        subcategory_edited_1 = function_init.clean_text(subcategory_name_1)
        database_edited = function_init.clean_text(database_name)
        wiki_level_0 = self.wiki_db[database_edited][database_edited][subcategory_edited_0][subcategory_edited_1]
        wiki_level_1 = self.wiki_db[database_edited][database_edited][subcategory_edited_0][subcategory_edited_1].subcategories
        these_pageids = function_init.pageids(subcategory_name_1)
        for pages in these_pageids:
            if 'Category:' not in function_init.find_title(pages):
                wiki_level_0.insert_one(function_init.get_page_contents(pages,
                                                                        database_name + ' -> ' + subcategory_name_0 +
                                                                       ' -> ' + subcategory_name_1))
            if 'Category:' in function_init.find_title(pages) :
                subcategory = function_init.find_title(pages)
                subcategory = subcategory.replace('Category:', database_name + ' -> ' + subcategory_name_0 + ' -> '
                                                 + subcategory_name_1 + ' -> ')
                wiki_level_1.insert_one(function_init.get_page_contents(pages, subcategory))
                
    def add_fourth_layer(self, database_name, subcategory_name_0, subcategory_name_1, subcategory_name_2):
        function_init = functions()
        subcategory_edited_0 = function_init.clean_text(subcategory_name_0)
        subcategory_edited_1 = function_init.clean_text(subcategory_name_1)
        subcategory_edited_2 = function_init.clean_text(subcategory_name_2)
        database_edited = function_init.clean_text(database_name)
        wiki_level_0 = self.wiki_db[database_edited][database_edited][subcategory_edited_0][subcategory_edited_1][subcategory_edited_2]
        wiki_level_1 = self.wiki_db[database_edited][database_edited][subcategory_edited_0][subcategory_edited_1][subcategory_edited_2].subcategories
        these_pageids = function_init.pageids(subcategory_name_2)
        for pages in these_pageids:
            if 'Category:' not in function_init.find_title(pages):
                wiki_level_0.insert_one(function_init.get_page_contents(pages,
                                                                        database_name + ' -> ' + subcategory_name_0 +
                                                                       ' -> ' + subcategory_name_1 + ' -> ' 
                                                                        + subcategory_name_2))
            if 'Category:' in function_init.find_title(pages) :
                subcategory = function_init.find_title(pages)
                subcategory = subcategory.replace('Category:', database_name + ' -> ' + subcategory_name_0 + ' -> '
                                                 + subcategory_name_1 + ' -> ' + subcategory_name_2 + ' -> ')
                wiki_level_1.insert_one(function_init.get_page_contents(pages, subcategory))

def search_term(search_term):
    
    list_dbs = []
    
    for names in client.database_names():
        for collections in client[names].collection_names():
            cursor_level = client[names][collections].find()
            db_level = list(cursor_level)
            df_level = pd.DataFrame(db_level)
            list_dbs.append(df_level)
    
    df = pd.concat(list_dbs)
    
    top_5_list = []
    final_df = pd.DataFrame()
    
    tfidf_vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
    initial_terms_encoded = tfidf_vectorizer.fit_transform(df['raw page text'])
    initial_terms_encoded_df = pd.DataFrame(initial_terms_encoded.toarray(), 
                                       index=df['raw page text'], 
                                       columns=tfidf_vectorizer.get_feature_names())
    
    search_terms_encoded = tfidf_vectorizer.transform([search_term])
    search_terms_encoded_df = pd.DataFrame(search_terms_encoded.toarray(), 
                                       index=[search_term], 
                                       columns=tfidf_vectorizer.get_feature_names())
    
    dtm_with_search_term = initial_terms_encoded_df.append(search_terms_encoded_df)
    n_components = 200
    SVD = TruncatedSVD(n_components)
    component_names = ["component_"+str(i+1) for i in range(n_components)]
    
    svd_matrix = SVD.fit_transform(dtm_with_search_term)
    svd_df = pd.DataFrame(svd_matrix, 
                      index=dtm_with_search_term.index, 
                      columns=component_names)
    
    search_term_svd_vector = svd_df.loc[search_terms_encoded_df.index]
    svd_df['cosine_sim'] = cosine_similarity(svd_df, search_term_svd_vector)
    top_5 = svd_df[['cosine_sim']].sort_values('cosine_sim', ascending=False).head(6)
    
    for page in range(1,6):
        top_5_list.append(df[df['raw page text'] == top_5.index[page]]['title'])
    
    final_df['title'] = pd.concat(top_5_list)
    return final_df

def add_new(database_name):
    database_edited = functions().clean_text(database_name)
    database = mongodb(database_name)
    database.drop_database()
    database.add_database()
    database.add_first_layer(database_name)
    for articles in list(client[database_edited].subcategories.find()):
    	try:
        	category = articles['title'].replace('Category:', '')
        	category_edited = functions().clean_text(category)
        	database.add_second_layer(database_name, category)
        except:
        	pass
        for categories in list(client[database_edited][database_edited][category_edited].subcategories.find()):
        	try:
        		new_subcat = categories['title'].replace('Category:', '')
           		new_subcat_edited = functions().clean_text(new_subcat)
           		database.add_third_layer(database_name, category, new_subcat)
           	except:
           		pass
           	for subcategories in list(client[database_edited][database_edited][category_edited][new_subcat_edited].subcategories.find()):
           		try:
           			final_subcat = subcategories['title'].replace('Category:', '')
           			database.add_fourth_layer(database_name, category, new_subcat, final_subcat)
           		except:
           			pass

quit = 0

while quit == 0:

	this_input = input("What would you like to do? \n 1. Search \n 2. Add \n 3. Remove \n 4. Show Categories \n 5. Show Subcategories \n 6. Quit \n")

	if this_input == 1:
		this_input = 0
		q = 0
		while q == 0:
			name = raw_input("Search (type 'q' to exit): ")
			if name == 'q':
				q = 1
				pass
			if name != 'q':
				print("Searching Wikipedia articles.")
				search = search_term(name)
				print(search)

	if this_input == 2:
		new_database = raw_input("Enter Wikipedia Category: ")
		print("Loading information into database. This may take a a while.")
		add_new(new_database)
		print("New database has been added")

	if this_input == 3:
		remove_database = raw_input("Wikipedia Category to be Removed: ")
		database = mongodb(remove_database)
		database.drop_database()

	if this_input == 4:
		print(client.database_names())

	if this_input == 5:
		category = raw_input("Show subcategories of: ")
		database = mongodb(category)
		print(database.collections())

	if this_input == 6:
		quit = 1
