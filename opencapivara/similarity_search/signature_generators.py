import pandas as pd
import numpy as np
import pydantic
from typing import Optional, Dict, List, Tuple
import re
from opencapivara.similarity_search.utils import text2canonical
from opencapivara.schemas import Signature

class SignatureGenerator():    
    def __init__(self, version:str='latest'):
        if version == 'latest':
            self.version = '0.1.0'
        else:
            self.version = version

    def vectorize_one(self, text: str) -> List[float]:
        raise NotImplementedError('Abstract method')

    def predict_one(self, text: str, name:str=None) -> Signature:
        signature = Signature(name=name, version=self.version, embedding=self.vectorize_one(text))
        return signature

    def predict(self, texts: List[str]) -> List[Signature]:
        return [self.predict_one(text) for text in texts]

########################
# Rule based
tags_all = {
        'persuassion': ['persuasive'],
        'charisma': ['charismatic', 'cheerfull'],
        'programming': ['coding', 'develop', 'developer'],
        'italian': [],
        'german': [],

        'Robotic Process Automation': ['RPA'],
        'uipath': [],
        'jobrouter': [],
        'API': [],
        'SAP': [],
        'python': [],
        'javascript': [],
        'script': ['powershell', 'bash'],

        'Teams': [],
        'Teams - Update photo': [],
        'linux': [],
        'O365': [],
        'Sharepoint': [],
        'Onedrive': [],
        'Fileservice': [],
        'Identity Management': ['IDM', 'user account'],
        'distribution group': ['dl', 'distribution list'],
        'End-Of-Life': [],
        'Active Directory': ['AD', 'AAD', 'azure AD', 'AD-Gruppe'],
        'GPO': ['group policy'],
        'EUMA': [],
        'Delete large files': [],
        'Install software': [],
        'Setup VPN': [],
        'End-Of-Life': [],

        'Microsoft Office - Excel': ['excel', 'MS spreedsheet'],

        'proofpoint': [],
        'exchange': [],
        'mailbox': [],

        'O365 Shared Mailbox': ['shared mailbox', 'group mailbox', 'geteiltes postfach', 'SHared', 'mailgroup', 'team email', 'shared email'],
        'O365 Shared Mailbox – Create': ['create', 'new shared', 'Criação', 'creation', 'schaffen', ],
        'O365 Shared Mailbox – Delete': ['delete', 'remove shared', 'remove all', 'removal', 'löschen', ],
        'O365 Shared Mailbox – Access': ['access', 'Add people', 'add user', 'include user', 'remove user', 'remove email', 'berechtigen', 'grant', ],
        'O365 Shared Mailbox – Update': ['merge', 'Change', 'configuration', 'configure', 'switch', 'modifications', ],

        'Windows Update': [],
        'Empirum': [],
        'AD': [],
        'Nexthink': ['next-think'],
        'driver': [],
        'firmware': [],
        'OKTA': ['MFA', 'multi factor authentication', '2FA', 'two factor authentication'],
        'MFA': [],
        'Intune': [],
        'Citrix': [],
        'SAP': [],
        'MobilePass': ['Mob. Pass'],
        'M365': ['Microsoft365', 'Microsoft 365', 'M. 365'],
        'M365 account': ['MS account', 'office account', '365 account', 'windows account'],

        'Configure device': [],
        #'Software operations': ['Troubleshoot', 'Install', 'uninstall', 'update', 'patch'],
        'purchase': ['order'],
        'ship': [],
        'stock': [],
        'recycle': [],
        'Replacement hardware': ['Replace hardware', 'hardware replace'],
        'peripheral': [],
        'dock station': [],
        'computing hardware': ['keyboard', 'mouse', 'touchscreen', 'monitor', 'audio & video', 'handheld', 'barcode scanner', ],
        'Manage device': [],
        'approved device': [],
        'printer': [],
        'scanner': [],
        'copier': [],

        'OS': ['Operating System', 'Operational System', 'Sistema Operacional'],
        'linux': [],
        'windows': [],
        'mac-os': [],
        'BIOS': [],
        'UEFI': [],

        'enrollment link': [],
        'phone settings': [],
        'connectivity': [],
        'Licensing': [],
        'vendor': [],

        'network device': ['switch', 'wireless', 'router', 'Access Point'],
        'firewall': [],
        'proxy': [],
        'Antivirus': [],
        'SSL': [],
        'network': ['networking', 'connectivity', 'communication', 'internet'],
        'domain controller': ['DC'],

        'AWS': ['amazon web services'],
        'Azure': ['Microsoft cloud'],
        'IBM cloud': [],
        'Google Cloud': ['GCP'],
        'public cloud': [],
        'private cloud': [],
        'on premise': [],
        
        'MPLS': ['Multiprotocol Label Switching'],
        'assets library': [],
        'Server': ['server rooms', 'servers'],
        'fileserver': ['file', 'fileservice', 'fyleserver', 'file server', 'FS'],
        'bare-metal': [],
        'hypervisor': ['virtual machine', 'virtual server', 'VM', 'EC2', 'virtualbox', 'HY', 'vmware'],
        'vmware': ['ESX', 'Vcenter', 'Vsphere'],
        'backup': [],        
        'VEEAM': [],
        'general usage': [],
        'database':  ['MSSQL', 'ORACLE', 'postgres', 'postgreSQL'],
        'failover architecture': [],
        'change requests': [],
        'Network company': ['Cisco', 'Aruba', 'HP networks', 'Aerohive', 'extreme Networks'],
        'WLAN': ['local area network', 'local net'],
        'local': ['on-site', 'Tangent', 'Bloomington', 'Kimberly', 'Shakopee'],
        'DHCP': ['Dynamic Host Configuration Protocol', 'IP configuration'],
        'DNS': ['Domain Name System'],
        'Certificate Authority': ['CA', 'certificates', 'cert', 'certs', "let's encript", 'HTTPS'],
        'MES': [],
        'HMI': ['Human-Machine Interface'],
        'SCADA': ['Supervisory control and data acquisition', 'data acquisition', 'data aquisition'],
        'LIMS': ['laboratory information management system'],
        'ACL': ['Audit Command Language'],
        'Nicelabel': ['label design', 'label printing'],
        'Harvest': ['time tracking'],
        'Sprite': [],


    #outlook
    #McAfee
    #BarTender
    #google earth
    #SICALC
    #wordpad
    #XFlow, X-Flow
    #google drive
    #Parallels
    #SAP
    #Adobe Acrobat
    #hobo
    #citrix
    #
    #Google Chrome
    #
    #Rstudio
    #Rsuite
    #visual studio code, vs code
    #Swift Console
    #
    #Datalogger Extech
    #Neoagro
}
class SignatureGeneratorRuleBased(SignatureGenerator):
    tags = tags_all #maps name to synonims/keyword
    def predict_one(self, text: str, name:str=None) -> Signature:
        signature = Signature(name=name, version=self.version)
        for tag in self.tags:
            # if the tag's keywords are present, add the tag to the signature
            # TODO: substitute `+' '` by regex `\b`
            # TODO: stop in first match
            if any([(' '+text2canonical(keyword)+' ' in ' '+text2canonical(text)+' ') for keyword in self.tags[tag] + [tag]]):
                signature.tags[tag] = 5 
                if re.search(rf'(not be|no|any|none) {text2canonical(tag)}', text2canonical(text)):
                    signature.tags[tag] = -10
                if re.search(rf'(fluent in|fluent|native|must|expert|expert in|great experience in) {text2canonical(tag)}', text2canonical(text)):
                    signature.tags[tag] = 10
                if re.search(rf'(learning|a little|basic) {text2canonical(tag)}', text2canonical(text)):
                    signature.tags[tag] = 3

        return signature

class SignatureGeneratorTFIDF(SignatureGenerator):
    def __init__(self, version: str = 'latest'):
        from joblib import load
        super().__init__(version=version)
        self.model = load('./assets/models/tf-idf.joblib')
        
    def vectorize_one(self, text: str) -> List[float]:
        text = text2canonical(text)
        embedding = self.model.transform([text]).toarray().tolist()[0]
        embedding = [float(e) for e in embedding]
        return embedding

#################### WORD2VEC
import numpy as np
import gensim.downloader as api
class SignatureGeneratorWord2vec(SignatureGenerator):
    '''
    Google News dataset (about 100 billion words)
    for English
    '''
    def __init__(self, version:str='latest'):
        print('Loading word2vec model...')
        self.wv = api.load('word2vec-google-news-300')
        print('Word2vec model loaded.')
        super().__init__(version = version)

    def vectorize_one(self, text: str) -> list:
        words = text.split()
        words = list(map(text2canonical, words))
        # remove out-of-vocabulary words
        words = [word for word in words if word in self.wv]
        if len(words) >= 1:
            return np.mean(self.wv[words], axis=0).tolist()
        else:
            return []



from gensim.models import KeyedVectors
class SignatureGeneratorWord2VecRetrained(SignatureGeneratorWord2vec):
    def __init__(self, version:str='latest'):
        super().__init__(version = version)
        self.wv = KeyedVectors.load("assets/models/word2vec_retrained.wordvectors", mmap='r')
        
class SignatureGeneratorDoc2VecRetrained(SignatureGenerator):
    def __init__(self, version: str = 'latest'):
        super().__init__(version = version)
        from gensim.models.doc2vec import Doc2Vec
        self.model = Doc2Vec.load('assets/models/doc2vec_retrained.wordvectors')

    def vectorize_one(self, text: str) -> List[float]:
        text = text2canonical(text)
        embedding = self.model.infer_vector([text]).tolist()
        return embedding


class SignatureGeneratorLDA(SignatureGenerator):
    def __init__(self, version: str = 'latest'):
        super().__init__(version = version)
        from gensim.corpora.dictionary import Dictionary
        from gensim.models import LdaModel
        self.dictionary = Dictionary.load('assets/models/lda-dictionary.model')
        self.lda = LdaModel.load('assets/models/lda.model')

    def vectorize_one(self, text: str) -> List[float]:
        text = text2canonical(text)
        text_bow = self.dictionary.doc2bow(text.split())
        embedding = [0.]*self.lda.num_topics
        for topic, value in self.lda[text_bow]:
            embedding[topic] = value
        #print('>>>>>>>>', type(embedding), len(embedding), embedding[:5])
        return embedding



################ BERT
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
import torch




class SignatureGeneratorBertMultilingual(SignatureGenerator):
    '''
    https://towardsdatascience.com/what-exactly-happens-when-we-fine-tune-bert-f5dc32885d76

    https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d

    TODO: I think it's ok, but I'm receiving the following warning:
    Some weights of the model checkpoint at bert-base-multilingual-cased were not used when initializing BertModel: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    --------------
    Maybe the NLU library could be a possibility

    About NLU usage:
    * https://medium.com/spark-nlp/1-line-to-albert-word-embeddings-with-nlu-in-python-1691bc048ed1
    * https://nlu.johnsnowlabs.com/docs/en/install
    * https://nlu.johnsnowlabs.com/#word-embeddings-bert
    * https://nlu.johnsnowlabs.com/#word-embeddings-elmo


    About NLU with retraining:
    * https://nlu.johnsnowlabs.com/docs/en/training
    * https://stackoverflow.com/questions/60089012/is-it-possible-to-load-a-trained-rasa-nlu-model-and-run-inference-to-get-embeddi


    !python3 -m pip install --upgrade pip
    !pip3 install bert-embedding
    from bert_embedding import BertEmbedding

    sentences = ['hello, let me sleep']
    model = BertEmbedding()
    result = model(sentences)
    result

    '''
    model_name: str = 'bert-base-multilingual-cased'
    def __init__(self, version:str='latest'):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        super().__init__(version = version)
        
    def predict_one(self, text: str, name:str=None) -> Signature:
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]
        embeddings = [float(e) for e in list(last_hidden_states[0][0])]
        signature = Signature(name=name, version=self.version, embedding=embeddings)
        return signature


class SignatureGeneratorSentenceBert(SignatureGenerator):
    model_name: str = 'all-MiniLM-L6-v2'
    def __init__(self, version:str='latest'):
        self.model = SentenceTransformer(self.model_name)
        super().__init__(version = version)
        
    def predict_one(self, text: str, name:str=None) -> Signature:
        embeddings = self.model.encode([text])
        signature = Signature(name=name, version=self.version, embedding=embeddings[0].tolist())
        return signature
    
class SignatureGeneratorSentenceBertMultilingual(SignatureGeneratorSentenceBert):
    model_name:str = 'distiluse-base-multilingual-cased-v1'
    
class SignatureGeneratorSentenceBertEnglish(SignatureGeneratorSentenceBert):
    '''
    All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs.
    https://huggingface.co/microsoft/mpnet-base
    '''
    model_name: str = 'all-mpnet-base-v2'

class SignatureGeneratorSentenceBertRetrained(SignatureGeneratorSentenceBert):
    model_name:str = './assets/models/sentence-bert-retrained'

class SignatureGeneratorRandom(SignatureGenerator):
    def predict_one(self, text: str, name:str=None) -> Signature:
        signature = Signature(name=name, version=self.version, embedding=np.random.random(size=10).tolist())
        return signature
