import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from spacy.tokens import DocBin
import warnings
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import geopandas as gpd
import collections

warnings.filterwarnings('ignore')

class searchEngine():
    def __init__(self, df, nlp):
        self.df = df
        self.nlp = nlp
    
    def spacyify(self,col_name,f_name='spacy_model_output'):
        texts = self.df[col_name].to_list()
        doc_bin = DocBin()
        for doc in tqdm(self.nlp.pipe(texts), total=len(texts)):
            doc_bin.add(doc)
        bytes_data = doc_bin.to_bytes()

        f = open(f'serialized_data/{f_name}','wb')
        f.write(bytes_data)
        f.close()
    
    def search(self, doc_bin_path, search_text, entries=5, context_size=2, mapping=False, **kwargs):
        if ' ' in search_text:
            search_vec = self.nlp(search_text)
        else:
            search_vec = self.nlp.vocab[search_text]
        
        bytes_file = open(doc_bin_path,'rb').read()
        doc_bin = DocBin().from_bytes(bytes_file)
        self.df['sent_docs'] =  pd.Series(doc_bin.get_docs(self.nlp.vocab))
  
        sim_score = self.df['sent_docs'].apply(lambda x: x.similarity(search_vec)).sort_values(ascending=False)[0:entries]
        sim_df = sim_score.reset_index().rename(columns={'index':'org_idx'})

        def createContext(org, context_size):
            context = self.df.sent_docs.iloc[org].text
            for i in range(context_size):
                if (i < len(self.df)) and (i > 0):
                    context = self.df.sent_docs.iloc[org-i].text + '\n' + context
                    context = context + '\n' + self.df.sent_docs.iloc[org+i].text
            return context

        def getLatLong(org, context_size):
            for i in range(context_size):
            ## default dicts are really useful for taskas like this
                places = collections.defaultdict(int)
                for ent in self.df.sent_docs.iloc[org].ents:
                    places[ent.text] += 1
                for ent in self.df.sent_docs.iloc[org-i].ents:
                    places[ent.text] += 1
                for ent in self.df.sent_docs.iloc[org+i].ents:
                    places[ent.text] += 1
                coords = []
                # filtering out noise from spaCy NER
                chars = ['Athos','Porthos','Aramis','Grimaud','Felton', 'Louise', 'Montalais', 'Mazarin']
                for place in places:
                    if (place != 'one') and (place != 'four') and (place != 'first') and (place != 'Roman') and (place != 'Le roi') and (place not in chars):
                    ## using geonames API to get coordinates
                        geoname = BeautifulSoup(requests.get(f'http://api.geonames.org/search?name_equals={place}&continentCode=EU&maxRows=10&username=pnadel').text).find('geoname')
                        if geoname != None:
                            lat = float(geoname.find('lat').get_text())
                            lng = float(geoname.find('lng').get_text())
                            coords.append((place,lat,lng,places[place]))
            return coords

        if mapping == True:
            sim_df['coords'] = sim_df['org_idx'].apply(lambda x: getLatLong(x, context_size))

        sim_df['context'] = sim_df['org_idx'].apply(lambda x: createContext(x, context_size))
        
        for key, value in kwargs.items():
            sim_df[key] = sim_df['org_idx'].apply(lambda x: self.df[value].iloc[x])
        return sim_df, search_text

    def europePlot(self, coords):
        if len(coords) > 0:
            ## coordinates in DF to allow for easier ploting
            coord_df = pd.DataFrame(coords,columns=['place','lat','lng','c'])

            ## open access geojson of Europe
            filename = "europe.geojson"
            file = open(filename)
            df = gpd.read_file(file)

            fig, ax = plt.subplots(figsize=(8,6))
            ## select countries to plot
            df[df['id'] == 'FR'].plot(ax=ax)
            df[df['id'] == 'ES'].plot(ax=ax)
            df[df['id'] == 'GB'].plot(ax=ax)
            df[df['id'] == 'BE'].plot(ax=ax)
            df[df['id'] == 'NL'].plot(ax=ax)
            df[df['id'] == 'IE'].plot(ax=ax)
            df[df['id'] == 'PT'].plot(ax=ax)
            df[df['id'] == 'AD'].plot(ax=ax)
            df[df['id'] == 'CH'].plot(ax=ax)

            ## using pandas built-in plot function
            coord_df.plot(x="lng",y='lat',kind='scatter', c='c',colormap="Reds", ax=ax)

            ## making labels
            for i in range(len(coord_df)):
                plt.text(coord_df.iloc[i].lng,coord_df.iloc[i].lat,f'{coord_df.iloc[i].place}')
            
            ax.grid(b=True, alpha=0.5)
            plt.show()

    def displaySearch(self, search_df, search_text, mapping=False, **kwargs):
        display(HTML(f'<h2>{search_text}</h2>'))
        display(HTML('<br>'))
        for i in range(len(search_df)):
            for key,value in kwargs.items():
                display(HTML(f'<small><i>{search_df[value].to_list()[i]}</i></small>'))
            display(HTML(f'<small>Similarity Score: {round(search_df.sent_docs.to_list()[i], 3)}</small>'))
            display(HTML(f'<p>{search_df.context.to_list()[i]}</p>'))
            if mapping == True:
                self.europePlot(search_df.coords.to_list()[i])
            display(HTML('<br>'))

    def searchWordOrPhrase(self, doc_bin_path,entries=5, context_size=2):
        search_term = input('Enter search term:')
        search = self.search(doc_bin_path,search_term,entries=entries, context_size=context_size)
        self.displaySearch(search[0],search[1])
